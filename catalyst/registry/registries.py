import logging

from catalyst.registry.registry import Registry
from catalyst.tools import settings

logger = logging.getLogger(__name__)


def _transforms_loader(r: Registry):
    from torch.jit.frontend import UnsupportedNodeError

    from catalyst.data.cv.transforms import torch as t

    r.add_from_module(t, prefix=["catalyst.", "C."])

    try:
        import albumentations as m

        r.add_from_module(m, prefix=["A.", "albu.", "albumentations."])

        from albumentations import pytorch as p

        r.add_from_module(p, prefix=["A.", "albu.", "albumentations."])

        from catalyst.data.cv.transforms import albumentations as t

        r.add_from_module(t, prefix=["catalyst.", "C."])
    except ImportError as ex:
        if settings.albumentations_required:
            logger.warning(
                "albumentations not available, to install albumentations, "
                "run `pip install albumentations`."
            )
            raise ex

    try:
        from kornia import augmentation as k

        r.add_from_module(k, prefix=["kornia."])
    except ImportError as ex:
        if settings.kornia_required:
            logger.warning(
                "kornia not available, to install kornia, "
                "run `pip install kornia`."
            )
            raise ex
    except UnsupportedNodeError as ex:
        logger.warning(
            "kornia has requirement torch>=1.5.0, probably you have"
            " an old version of torch which is incompatible.\n"
            "To update pytorch, run `pip install -U 'torch>=1.5.0'`."
        )
        if settings.kornia_required:
            raise ex


TRANSFORM = Registry("transform")
TRANSFORM.late_add(_transforms_loader)
Transform = TRANSFORM.add


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


SAMPLER = Registry("sampler")
SAMPLER.late_add(_samplers_loader)
Sampler = SAMPLER.add


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


GRAD_CLIPPER = Registry("func", default_meta_factory=_GradClipperWrap)
GRAD_CLIPPER.late_add(_grad_clip_loader)


def _modules_loader(r: Registry):
    from catalyst.contrib.nn import modules as m

    r.add_from_module(m)


MODULE = Registry("module")
MODULE.late_add(_modules_loader)
Module = MODULE.add


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


MODEL = Registry("model")
MODEL.late_add(_model_loader)
Model = MODEL.add


def _criterion_loader(r: Registry):
    from catalyst.contrib.nn import criterion as m

    r.add_from_module(m)


CRITERION = Registry("criterion")
CRITERION.late_add(_criterion_loader)
Criterion = CRITERION.add


def _optimizers_loader(r: Registry):
    from catalyst.contrib.nn import optimizers as m

    r.add_from_module(m)


OPTIMIZER = Registry("optimizer")
OPTIMIZER.late_add(_optimizers_loader)
Optimizer = OPTIMIZER.add


def _schedulers_loader(r: Registry):
    from catalyst.contrib.nn import schedulers as m

    r.add_from_module(m)


SCHEDULER = Registry("scheduler")
SCHEDULER.late_add(_schedulers_loader)
Scheduler = SCHEDULER.add


EXPERIMENT = Registry("experiment")
Experiment = EXPERIMENT.add

RUNNER = Registry("runner")
Runner = RUNNER.add


def _callbacks_loader(r: Registry):
    from catalyst.core import callbacks as m

    r.add_from_module(m)

    from catalyst.dl import callbacks as m  # noqa: WPS347

    r.add_from_module(m)

    from catalyst.contrib.dl import callbacks as m  # noqa: WPS347

    r.add_from_module(m)


CALLBACK = Registry("callback")
CALLBACK.late_add(_callbacks_loader)
Callback = CALLBACK.add


# backward compatibility
CALLBACKS = CALLBACK
CRITERIONS = CRITERION
EXPERIMENTS = EXPERIMENT
GRAD_CLIPPERS = GRAD_CLIPPER
MODELS = MODEL
MODULES = MODULE
OPTIMIZERS = OPTIMIZER
RUNNERS = RUNNER
SAMPLERS = SAMPLER
SCHEDULERS = SCHEDULER
TRANSFORMS = TRANSFORM


__all__ = [
    "Callback",
    "CALLBACK",
    "CALLBACKS",
    "Criterion",
    "CRITERION",
    "CRITERIONS",
    "Experiment",
    "EXPERIMENT",
    "EXPERIMENTS",
    "GRAD_CLIPPER",
    "GRAD_CLIPPERS",
    "Model",
    "MODEL",
    "MODELS",
    "Module",
    "MODULE",
    "MODULES",
    "Optimizer",
    "OPTIMIZER",
    "OPTIMIZERS",
    "Runner",
    "RUNNER",
    "RUNNERS",
    "Sampler",
    "SAMPLER",
    "SAMPLERS",
    "Scheduler",
    "SCHEDULER",
    "SCHEDULERS",
    "Transform",
    "TRANSFORM",
    "TRANSFORMS",
]
