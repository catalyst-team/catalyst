from catalyst.contrib.registry import *
from catalyst.core.registry import *
from catalyst.utils.tools.registry import Registry


def _callbacks_loader(r: Registry):
    from catalyst.dl import callbacks as m
    r.add_from_module(m)


CALLBACKS.late_add(_callbacks_loader)


__all__ = [
    "Callback",
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Module",
    "Model",
    "Sampler",
    "Transform",
    "CALLBACKS",
    "CRITERIONS",
    "GRAD_CLIPPERS",
    "MODELS",
    "MODULES",
    "OPTIMIZERS",
    "SAMPLERS",
    "SCHEDULERS",
    "TRANSFORMS",
]
