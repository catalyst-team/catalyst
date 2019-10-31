from catalyst.contrib.registry import (
    Criterion, CRITERIONS, GRAD_CLIPPERS, Model, MODELS, Module, MODULES,
    Optimizer, OPTIMIZERS, Scheduler, SCHEDULERS
)
from ..utils.registry import Registry


def _callbacks_loader(r: Registry):
    from catalyst.dl import callbacks as m
    r.add_from_module(m)


CALLBACKS = Registry("callback")
CALLBACKS.late_add(_callbacks_loader)
Callback = CALLBACKS.add

__all__ = [
    "Criterion", "Optimizer", "Scheduler", "Callback", "Module", "Model",
    "MODULES", "CRITERIONS", "OPTIMIZERS", "SCHEDULERS", "GRAD_CLIPPERS",
    "MODELS"
]
