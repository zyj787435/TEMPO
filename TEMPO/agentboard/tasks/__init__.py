from .alfworld import Evalalfworld
from .alfworld_dqn import EvalalfworldDQN
from .alfworld_rl_v1 import EvalalfworldRLv1
from .alfworld_rl_v2 import EvalalfworldRLv2
from .scienceworld import EvalScienceworld
from .scienceworld_dqn import EvalScienceworldDQN
from .alfworld_hierarchical import EvalalfworldHierarchical
from .scienceworld_hierarchical import EvalScienceworldHierarchical
from .webshop_hierarchical import EvalWebshopHierarchical
from .webshop_flat import EvalWebshopFlat
from .tool import EvalTool

from common.registry import registry

__all__ = [
    "Evalalfworld",
    "EvalBabyai",
    "EvalPddl",
    # "EvalWebBrowse",
    "EvalWebshop",
    "EvalJericho",
    "EvalTool",
    "EvalWebshop",
    "EvalScienceworld"
]


def load_task(name, run_config, llm_config, agent_config, env_config, llm=None):
    # Lazy imports for tasks that need 'llm' module on sys.path
    if name == "pddl" and "pddl" not in registry.list_tasks():
        from .pddl import EvalPddl
    if name == "pddl_hierarchical" and "pddl_hierarchical" not in registry.list_tasks():
        from .pddl_hierarchical import EvalPddlHierarchical
    task = registry.get_task_class(name).from_config(run_config, llm_config, agent_config, env_config, llm=llm)

    return task

