from catalyst.contrib.registry import (
    MODULES, CRITERIONS, OPTIMIZERS, SCHEDULERS, GRAD_CLIPPERS, Module,
    Optimizer, Scheduler, Criterion
)
from ..utils.registry import Registry


def _agents_late_add(r: Registry):
    from . import agents as m
    r.add_from_module(m)


AGENTS = Registry("agent")
AGENTS.late_add(_agents_late_add)
Agent = AGENTS.add


def _algorithms_late_add(r: Registry):
    from .offpolicy import algorithms as m
    r.add_from_module(m)


ALGORITHMS = Registry("algorithm")
ALGORITHMS.late_add(_algorithms_late_add)
Algorithm = ALGORITHMS.add


def _env_late_add(r: Registry):
    from . import environments as m
    r.add_from_module(m)


ENVIRONMENTS = Registry("environment")
ENVIRONMENTS.late_add(_env_late_add)
Environment = ENVIRONMENTS.add


def _exploration_late_add(r: Registry):
    from .offpolicy import exploration as m
    r.add_from_module(m)


EXPLORATION = Registry("exploration")
EXPLORATION.late_add(_exploration_late_add)
Exploration = EXPLORATION.add

__all__ = [
    "Agent", "Algorithm", "Environment", "Exploration", "AGENTS", "ALGORITHMS",
    "ENVIRONMENTS", "EXPLORATION", "Module", "Criterion", "Optimizer",
    "Scheduler", "MODULES", "CRITERIONS", "OPTIMIZERS", "SCHEDULERS",
    "GRAD_CLIPPERS"
]
