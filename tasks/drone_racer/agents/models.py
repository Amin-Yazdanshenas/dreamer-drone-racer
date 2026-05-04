import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# Must match drone_racer_env_cfg.py camera resolution and PolicyCfg obs terms
IMAGE_H = 64
IMAGE_W = 64
IMAGE_DIM = IMAGE_H * IMAGE_W  # 4096 — flat binary gate mask from gate_mask()
IMU_DIM = 7                    # imu_ang_vel(3) + imu_att(4)


class CNNPolicy(GaussianMixin, Model):
    """FPV policy: CNN encoder over grayscale image, concatenated with IMU, then MLP.

    Architecture (64×64 input):
        Conv(1→16, 5×5, s=2) → 32×32
        Conv(16→32, 3×3, s=2) → 16×16
        Conv(32→64, 3×3, s=2) → 8×8
        Conv(64→64, 3×3, s=2) → 4×4  → flatten → 1024
        Linear(1024→256)
        cat(image_feat[256], imu[7]) → 263
        Linear(263→256) → Linear(256→256) → Linear(256→actions)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
    ):
        # In skrl 1.4.x all __init__ args are keyword-only; Model must be called before mixins
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # → 32×32
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # → 16×16
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → 8×8
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # → 4×4
            nn.ELU(),
        )
        cnn_flat = 64 * 4 * 4  # 1024

        self.cnn_proj = nn.Sequential(nn.Linear(cnn_flat, 256), nn.ELU())

        self.mlp = nn.Sequential(
            nn.Linear(256 + IMU_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_param = nn.Parameter(torch.full((self.num_actions,), initial_log_std))

    def compute(self, inputs, role):
        x = inputs["observations"]  # (N, IMAGE_DIM + IMU_DIM) — from OBSERVATIONS (policy obs group)

        img = x[:, :IMAGE_DIM].reshape(-1, 1, IMAGE_H, IMAGE_W)
        imu = x[:, IMAGE_DIM:]

        feat = self.cnn(img).reshape(x.size(0), -1)
        feat = self.cnn_proj(feat)

        h = self.mlp(torch.cat([feat, imu], dim=-1))
        mean = self.mean_layer(h)

        return mean, {"log_std": self.log_std_param}


class MLPCritic(DeterministicMixin, Model):
    """MLP value network for privileged ground-truth state (STATES, 20-dim)."""

    def __init__(self, state_space, action_space, device, clip_actions: bool = False):
        # state_space → STATES (critic obs group); observation_space not used by this model
        Model.__init__(self, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_states, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}  # reads STATES (critic obs group)


class MonoRacePolicy(GaussianMixin, Model):
    """Compact MLP policy for MonoRace perception pipeline.

    Architecture: obs_dim → 128 → ELU → 128 → ELU → 128 → ELU → actions
    Designed for 26-dim (no frame stacking) or 44-dim (K=3 stack) observations.
    ~37K parameters vs CNNPolicy ~330K.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        obs_dim = self.num_observations
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_param = nn.Parameter(torch.full((self.num_actions,), initial_log_std))

        perc_dim = obs_dim - 17  # remaining dims after lin_vel(3)+ang_vel(3)+quat(4)+action(4)+target(3)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[MonoRacePolicy] obs_dim={obs_dim} (gate_perc[0:{perc_dim}] | lin_vel[{perc_dim}:{perc_dim+3}] | "
            f"ang_vel[{perc_dim+3}:{perc_dim+6}] | quat[{perc_dim+6}:{perc_dim+10}] | "
            f"prev_action[{perc_dim+10}:{perc_dim+14}] | target_pos_b[{perc_dim+14}:{obs_dim}]), "
            f"actions={self.num_actions}, total_params={total_params:,}"
        )

    def compute(self, inputs, role):
        h = self.net(inputs["observations"])
        return self.mean_layer(h), {"log_std": self.log_std_param}
