"""Sequence replay buffer for R2-Dreamer — stores episodes, samples fixed-length chunks.

Sample output format: (B, T, ...) tensors — R2-Dreamer convention.
Image stays as uint8; agent converts to float internally.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class EpisodeBuffer:
    """Stores a single growing episode (one parallel env)."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[str, List] = {}

    def append(self, step: Dict[str, np.ndarray]) -> None:
        for k, v in step.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

    def flush(self) -> Optional[Dict[str, np.ndarray]]:
        """Return episode as numpy arrays and clear internal buffers."""
        if not self._data:
            return None
        episode = {k: np.stack(v, axis=0) for k, v in self._data.items()}
        self._data = {}
        return episode

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))


class SequenceReplayBuffer:
    """Ring-buffer of complete episodes; samples fixed-length (seq_len) chunks.

    Observation keys stored:
        image   : (H, W, C) uint8
        state   : (state_dim,) float32
        action  : (action_dim,) float32
        reward  : () float32
        is_first: () bool
        is_last : () bool

    Sample output tensors have shape (batch_size, seq_len, *feature_dims).
    Image is kept as uint8 — agent converts to float/255 internally.
    """

    def __init__(self, capacity: int, seq_len: int, num_envs: int,
                 device: str = "cpu"):
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_envs = num_envs
        self.device = device

        # One episode buffer per parallel env
        self._ep_bufs: List[EpisodeBuffer] = [EpisodeBuffer() for _ in range(num_envs)]

        # Completed episodes stored as a deque; evict oldest when over capacity
        self._episodes: deque = deque()
        self._total_steps: int = 0

        # Cached sampling state — rebuilt only when self._episodes is mutated. Without this,
        # sample() rebuilt list(deque), lengths array, and weights vector on every batch — O(N_eps)
        # work per draw. With replay_capacity=2M and avg ep ~150 steps, N_eps ≈ 13K, so this used to
        # dominate post-warmup wall-time. The dirty flag is flipped in _store_episode / eviction.
        self._sample_cache_dirty: bool = True
        self._episodes_list: list = []
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Add transitions
    # ------------------------------------------------------------------

    def add(self, obs_dict: Dict[str, torch.Tensor],
            actions: torch.Tensor,
            rewards: torch.Tensor,
            is_first: torch.Tensor,
            is_last: torch.Tensor) -> None:
        """Add one vectorised step (N envs) to the buffer.

        All tensors on any device; converted to CPU numpy here for storage.
        """
        N = actions.shape[0]
        image_np = obs_dict["image"].cpu().numpy()              # (N, H, W, C) uint8
        state_np = obs_dict["state"].cpu().float().numpy()      # (N, D) float32
        act_np = actions.cpu().float().numpy()                  # (N, A) float32
        rew_np = rewards.cpu().float().numpy()                  # (N,)
        first_np = is_first.cpu().bool().numpy()                # (N,)
        last_np = is_last.cpu().bool().numpy()                  # (N,)

        for i in range(N):
            step = {
                "image": image_np[i],
                "state": state_np[i],
                "action": act_np[i],
                "reward": rew_np[i],
                "is_first": first_np[i],
                "is_last": last_np[i],
            }
            self._ep_bufs[i].append(step)

            if last_np[i]:
                # Pad short episodes to seq_len by repeating the terminal frame so crash/early-
                # termination episodes still contribute their reward signal to the buffer.
                # Without this, early-training crashes (very common) silently disappeared and
                # the reward head never saw the terminal/crash signal — return/scale stuck at 1.0.
                cur_len = len(self._ep_bufs[i])
                if 0 < cur_len < self.seq_len:
                    pad = self.seq_len - cur_len
                    data = self._ep_bufs[i]._data
                    for k, lst in data.items():
                        last = lst[-1]
                        for _ in range(pad):
                            if k == "reward":
                                lst.append(np.zeros_like(last))
                            elif k == "is_first":
                                lst.append(np.zeros_like(last))
                            elif k == "is_last":
                                lst.append(np.ones_like(last))
                            else:
                                lst.append(last.copy() if hasattr(last, "copy") else last)
                ep = self._ep_bufs[i].flush()
                if ep is not None:
                    self._store_episode(ep)

    def _store_episode(self, episode: Dict[str, np.ndarray]) -> None:
        ep_len = len(episode["reward"])
        self._episodes.append(episode)
        self._total_steps += ep_len
        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and self._episodes:
            old = self._episodes.popleft()
            self._total_steps -= len(old["reward"])
        self._sample_cache_dirty = True

    # ------------------------------------------------------------------
    # Sample — returns (B, T, ...) format (R2-Dreamer convention)
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of (seq_len,) sequences.

        Returns dict with tensors of shape (batch_size, seq_len, *feature).
        image stays as uint8 — agent converts to float.
        Returns None if not enough data.
        """
        if not self._can_sample(batch_size):
            return None

        # Rebuild sample-side cache only when episode set has changed since last sample.
        if self._sample_cache_dirty or self._weights is None:
            self._episodes_list = list(self._episodes)
            lengths = np.array(
                [len(ep["reward"]) for ep in self._episodes_list], dtype=np.float32
            )
            total = float(lengths.sum())
            self._weights = (lengths / total) if total > 0 else None
            self._sample_cache_dirty = False

        episodes = self._episodes_list
        weights = self._weights

        # Batched draw of episode indices — one numpy call instead of batch_size separate ones.
        ep_indices = np.random.choice(len(episodes), size=batch_size, p=weights)

        sequences = []
        for ep_idx in ep_indices:
            ep = episodes[int(ep_idx)]
            ep_len = len(ep["reward"])
            max_start = ep_len - self.seq_len
            start = np.random.randint(0, max_start + 1)
            seq = {k: v[start: start + self.seq_len] for k, v in ep.items()}
            sequences.append(seq)

        # Stack: (batch_size, seq_len, *dims) — R2-Dreamer (B, T, ...) convention
        batch: Dict[str, torch.Tensor] = {}
        for k in sequences[0]:
            arr = np.stack([s[k] for s in sequences], axis=0)  # (B, T, ...)
            t = torch.from_numpy(arr).to(self.device)
            # image stays uint8 — agent preprocesses
            batch[k] = t

        return batch

    def _can_sample(self, batch_size: int) -> bool:
        if not self._episodes:
            return False
        valid = sum(1 for ep in self._episodes if len(ep["reward"]) >= self.seq_len)
        return valid >= batch_size

    def __len__(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)
