from catalyst.contrib.registry import (
    Criterion, CRITERIONS, GRAD_CLIPPERS, Model, MODELS, Module, MODULES,
    Optimizer, OPTIMIZERS, Sampler, SAMPLERS, Scheduler, SCHEDULERS
)
from catalyst.utils.registry import Registry
from catalyst.core.registry import CALLBACKS, Callback


def _callbacks_loader(r: Registry):
    from catalyst.dl import callbacks as m
    r.add_from_module(m)


CALLBACKS.late_add(_callbacks_loader)

__all__ = [
    "Criterion", "Optimizer", "Scheduler", "Callback", "Module", "Model",
    "Sampler", "MODULES", "CRITERIONS", "OPTIMIZERS", "SCHEDULERS",
    "GRAD_CLIPPERS", "MODELS", "SAMPLERS"
]
