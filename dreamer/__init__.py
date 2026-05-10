"""R2-Dreamer / NE-Dreamer — PyTorch world-model RL for Isaac Lab drone racing."""

from .agent import DreamerConfig, DreamerV3Agent
from .ne_agent import NEDreamerV3Agent

# env_wrapper requires Isaac Sim — import lazily to allow unit testing without sim
try:
    from .env_wrapper import DreamerIsaacEnvWrapper
except (ImportError, ModuleNotFoundError):
    DreamerIsaacEnvWrapper = None  # type: ignore[assignment,misc]

__all__ = ["DreamerV3Agent", "NEDreamerV3Agent", "DreamerConfig", "DreamerIsaacEnvWrapper"]
