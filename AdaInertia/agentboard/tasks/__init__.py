from .alfworld import Evalalfworld
from .alfworld_rl_v2 import EvalalfworldRLv2
from .scienceworld import EvalScienceworld
from .tool import EvalTool

from common.registry import registry

__all__ = [
    "Evalalfworld",
    "EvalalfworldRLv2",
    "EvalScienceworld",
    "EvalTool",
]


def load_task(name, run_config, llm_config, agent_config, env_config, llm=None):
    task = registry.get_task_class(name).from_config(run_config, llm_config, agent_config, env_config, llm=llm)
    
    return task

