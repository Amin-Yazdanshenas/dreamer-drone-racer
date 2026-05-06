"""Sequence replay buffer for DreamerV3 — stores episodes, samples fixed-length chunks."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

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
    """

    def __init__(self, capacity: int, seq_len: int, num_envs: int,
                 device: str = "cpu"):
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_envs = num_envs
        self.device = device

        # One episode buffer per parallel env (collects in-progress episode)
        self._ep_bufs: List[EpisodeBuffer] = [EpisodeBuffer() for _ in range(num_envs)]

        # Completed episodes stored as a deque; evict oldest when over capacity
        self._episodes: deque = deque()
        self._total_steps: int = 0

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
        image_np = obs_dict["image"].cpu().numpy()    # (N, H, W, C) uint8
        state_np = obs_dict["state"].cpu().float().numpy()   # (N, D) float32
        act_np = actions.cpu().float().numpy()               # (N, A) float32
        rew_np = rewards.cpu().float().numpy()               # (N,)
        first_np = is_first.cpu().bool().numpy()             # (N,)
        last_np = is_last.cpu().bool().numpy()               # (N,)

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
                if len(self._ep_bufs[i]) >= self.seq_len:
                    ep = self._ep_bufs[i].flush()
                    if ep is not None:
                        self._store_episode(ep)
                else:
                    # Too short to sample from — discard, but MUST clear the buffer so the
                    # next episode doesn't inherit this episode's data (cross-episode contamination).
                    self._ep_bufs[i].flush()

    def _store_episode(self, episode: Dict[str, np.ndarray]) -> None:
        ep_len = len(episode["reward"])
        self._episodes.append(episode)
        self._total_steps += ep_len
        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and self._episodes:
            old = self._episodes.popleft()
            self._total_steps -= len(old["reward"])

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of (seq_len,) sequences.

        Returns dict with tensors of shape (batch_size, seq_len, *feature),
        transposed for RSSM to (seq_len, batch_size, *feature).
        Returns None if not enough data.
        """
        if not self._can_sample(batch_size):
            return None

        sequences = []
        episodes = list(self._episodes)
        # Weighted sampling by episode length (longer episodes more likely to be sampled)
        lengths = np.array([len(ep["reward"]) for ep in episodes], dtype=np.float32)
        weights = lengths / lengths.sum()

        for _ in range(batch_size):
            ep_idx = np.random.choice(len(episodes), p=weights)
            ep = episodes[ep_idx]
            ep_len = len(ep["reward"])
            max_start = ep_len - self.seq_len
            start = np.random.randint(0, max_start + 1)
            seq = {k: v[start: start + self.seq_len] for k, v in ep.items()}
            sequences.append(seq)

        # Stack: (batch_size, seq_len, *dims) → transpose to (seq_len, batch_size, *dims)
        batch: Dict[str, torch.Tensor] = {}
        for k in sequences[0]:
            arr = np.stack([s[k] for s in sequences], axis=0)   # (B, T, ...)
            t = torch.from_numpy(arr).to(self.device)
            # Move seq_len axis first: (B, T, ...) → (T, B, ...)
            dims = list(range(t.ndim))
            dims[0], dims[1] = dims[1], dims[0]
            batch[k] = t.permute(*dims).contiguous()
            # Convert image to float [0,1] for the model
            if k == "image":
                batch[k] = batch[k].float() / 255.0

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
