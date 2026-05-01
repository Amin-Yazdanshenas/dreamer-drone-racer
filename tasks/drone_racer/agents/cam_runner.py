import copy

from skrl.utils.runner.torch import Runner

from .models import CNNPolicy, MLPCritic

# Task IDs that use the camera-based asymmetric actor-critic setup
CAM_TASKS = {
    "Isaac-Drone-Racer-v0",
    "Isaac-Drone-Racer-Play-v0",
}


class CamRunner(Runner):
    """Runner subclass that replaces YAML model instantiation with CNN policy + MLP critic.

    Everything else (agent config mapping, preprocessors, trainer, learning-rate
    scheduler) is handled by the parent Runner unchanged.
    """

    def _generate_models(self, env, cfg):
        device = env.device
        policy_cfg = cfg.get("models", {}).get("policy", {})

        policy = CNNPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            clip_actions=policy_cfg.get("clip_actions", False),
            clip_log_std=policy_cfg.get("clip_log_std", True),
            min_log_std=policy_cfg.get("min_log_std", -20.0),
            max_log_std=policy_cfg.get("max_log_std", 2.0),
            initial_log_std=policy_cfg.get("initial_log_std", 0.0),
        )

        # Critic reads STATES (critic obs group → env.state_space); pass as state_space not observation_space
        state_space = (
            env.state_space
            if hasattr(env, "state_space") and env.state_space is not None
            else env.observation_space
        )
        value = MLPCritic(
            state_space=state_space,
            action_space=env.action_space,
            device=device,
        )

        models = {"policy": policy, "value": value}
        for role, model in models.items():
            model.init_state_dict(role=role)

        # skrl Runner._generate_agent indexes models by agent_id ("agent" for single-agent)
        return {"agent": models}
