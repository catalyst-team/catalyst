"""
catalyst.dl subpackage registries
"""

from catalyst.utils.registry import Registry


class _GradClipperWrap:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        self.fn(x, *self.args, **self.kwargs)


def _grad_clip_loader(r: Registry):
    from torch.nn.utils import clip_grad as m
    r.add_from_module(m)


GRAD_CLIPPERS = Registry("func", default_meta_factory=_GradClipperWrap)
GRAD_CLIPPERS.late_add(_grad_clip_loader)


def _criterion_loader(r: Registry):
    from catalyst.contrib import criterion as m
    r.add_from_module(m)


CRITERIONS = Registry("criterion")
CRITERIONS.late_add(_criterion_loader)
Criterion = CRITERIONS.add


def _model_loader(r: Registry):
    from catalyst.contrib import models as m
    r.add_from_module(m)


MODELS = Registry("model")
MODELS.late_add(_model_loader)
Model = MODELS.add


def _modules_loader(r: Registry):
    from catalyst.contrib import modules as m
    r.add_from_module(m)


MODULES = Registry("module")
MODULES.late_add(_modules_loader)
Module = MODULES.add


def _optimizers_loader(r: Registry):
    from catalyst.contrib import optimizers as m
    r.add_from_module(m)


OPTIMIZERS = Registry("optimizer")
OPTIMIZERS.late_add(_optimizers_loader)
Optimizer = OPTIMIZERS.add


def _schedulers_loader(r: Registry):
    from catalyst.contrib import schedulers as m
    r.add_from_module(m)


SCHEDULERS = Registry("scheduler")
SCHEDULERS.late_add(_schedulers_loader)
Scheduler = SCHEDULERS.add

__all__ = ["Criterion", "Model", "Module", "Optimizer", "Scheduler"]
