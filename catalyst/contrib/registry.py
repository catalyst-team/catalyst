"""
catalyst subpackage registries
"""
import logging

from catalyst.tools import settings
from catalyst.tools.registry import Registry

logger = logging.getLogger(__name__)


def _transforms_loader(r: Registry):
    try:
        import albumentations as m

        r.add_from_module(m, prefix=["A.", "albu.", "albumentations."])

        from albumentations import pytorch as p

        r.add_from_module(p, prefix=["A.", "albu.", "albumentations."])

        from catalyst.contrib.data.cv import transforms as t

        r.add_from_module(t, prefix=["catalyst.", "C."])
    except ImportError as ex:
        if settings.albumentations_required:
            logger.warning(
                "albumentations not available, to install albumentations, "
                "run `pip install albumentations`."
            )
            raise ex


TRANSFORMS = Registry("transform")
TRANSFORMS.late_add(_transforms_loader)
Transform = TRANSFORMS.add


def _samplers_loader(r: Registry):
    from torch.utils.data import sampler as s

    factories = {
        k: v
        for k, v in s.__dict__.items()
        if "Sampler" in k and k != "Sampler"
    }
    r.add(**factories)
    from catalyst.data import sampler

    r.add_from_module(sampler)


SAMPLERS = Registry("sampler")
SAMPLERS.late_add(_samplers_loader)
Sampler = SAMPLERS.add


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


def _modules_loader(r: Registry):
    from catalyst.contrib.nn import modules as m

    r.add_from_module(m)


MODULES = Registry("module")
MODULES.late_add(_modules_loader)
Module = MODULES.add


def _model_loader(r: Registry):
    from catalyst.contrib import models as m

    r.add_from_module(m)

    try:
        import segmentation_models_pytorch as smp

        r.add_from_module(smp, prefix="smp.")
    except ImportError as ex:
        if settings.segmentation_models_required:
            logger.warning(
                "segmentation_models_pytorch not available,"
                " to install segmentation_models_pytorch,"
                " run `pip install segmentation-models-pytorch`."
            )
            raise ex


MODELS = Registry("model")
MODELS.late_add(_model_loader)
Model = MODELS.add


def _criterion_loader(r: Registry):
    from catalyst.contrib.nn import criterion as m

    r.add_from_module(m)


CRITERIONS = Registry("criterion")
CRITERIONS.late_add(_criterion_loader)
Criterion = CRITERIONS.add


def _optimizers_loader(r: Registry):
    from catalyst.contrib.nn import optimizers as m

    r.add_from_module(m)


OPTIMIZERS = Registry("optimizer")
OPTIMIZERS.late_add(_optimizers_loader)
Optimizer = OPTIMIZERS.add


def _schedulers_loader(r: Registry):
    from catalyst.contrib.nn import schedulers as m

    r.add_from_module(m)


SCHEDULERS = Registry("scheduler")
SCHEDULERS.late_add(_schedulers_loader)
Scheduler = SCHEDULERS.add


EXPERIMENTS = Registry("experiment")
Experiment = EXPERIMENTS.add


__all__ = [
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Module",
    "Model",
    "Sampler",
    "Transform",
    "Experiment",
    "CRITERIONS",
    "GRAD_CLIPPERS",
    "MODELS",
    "MODULES",
    "OPTIMIZERS",
    "SAMPLERS",
    "SCHEDULERS",
    "TRANSFORMS",
    "EXPERIMENTS",
]
