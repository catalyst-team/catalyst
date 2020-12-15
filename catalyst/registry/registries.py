import logging

from catalyst.registry.subregistry import SubRegistry as Registry
from catalyst.registry.registry import Registry as MegaRegistry
from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)


def _transforms_loader(r: Registry):
    from torch.jit.frontend import UnsupportedNodeError

    from catalyst.contrib.data.cv.transforms import torch as t

    r.add_from_module(t, prefix=["catalyst.", "C."])

    try:
        import albumentations as m

        r.add_from_module(m, prefix=["A.", "albu.", "albumentations."])

        from albumentations import pytorch as p

        r.add_from_module(p, prefix=["A.", "albu.", "albumentations."])

        from catalyst.contrib.data.cv.transforms import albumentations as t

        r.add_from_module(t, prefix=["catalyst.", "C."])
    except ImportError as ex:
        if SETTINGS.albumentations_required:
            logger.warning(
                "albumentations not available, to install albumentations, "
                "run `pip install albumentations`."
            )
            raise ex

    try:
        from kornia import augmentation as k

        r.add_from_module(k, prefix=["kornia."])

        from catalyst.contrib.data.cv.transforms import kornia as t

        r.add_from_module(t, prefix=["catalyst.", "C."])
    except ImportError as ex:
        if SETTINGS.kornia_required:
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
        if SETTINGS.kornia_required:
            raise ex


# TRANSFORM = Registry("transform")
# TRANSFORM.late_add(_transforms_loader)
# Transform = TRANSFORM.add
MegaRegistry_ = MegaRegistry()
MegaRegistry_['transform'].late_add(_transforms_loader)
TRANSFORM = MegaRegistry_['transform']
Transform = MegaRegistry_['transform'].add


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


# SAMPLER = Registry("sampler")
# SAMPLER.late_add(_samplers_loader)
# Sampler = SAMPLER.add
MegaRegistry_['sampler'].late_add(_samplers_loader)
SAMPLER = MegaRegistry_['sampler']
Sampler = MegaRegistry_['sampler'].add


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


# # @TODO: why func? should be renamed
# GRAD_CLIPPER = Registry("func", default_meta_factory=_GradClipperWrap)
# GRAD_CLIPPER.late_add(_grad_clip_loader)
MegaRegistry_.add_subregistry('func', default_meta_factory=_GradClipperWrap).late_add(_grad_clip_loader)
GRAD_CLIPPER = MegaRegistry_['func']


def _modules_loader(r: Registry):
    from catalyst.contrib.nn import modules as m

    r.add_from_module(m)


# MODULE = Registry("module")
# MODULE.late_add(_modules_loader)
# Module = MODULE.add
MegaRegistry_['module'].late_add(_modules_loader)
MODULE = MegaRegistry_['module']
Module = MegaRegistry_['module'].add


def _model_loader(r: Registry):
    from catalyst.contrib import models as m

    r.add_from_module(m)

    try:
        import segmentation_models_pytorch as smp

        r.add_from_module(smp, prefix="smp.")
    except ImportError as ex:
        if SETTINGS.segmentation_models_required:
            logger.warning(
                "segmentation_models_pytorch not available,"
                " to install segmentation_models_pytorch,"
                " run `pip install segmentation-models-pytorch`."
            )
            raise ex


# MODEL = Registry("model")
# MODEL.late_add(_model_loader)
# Model = MODEL.add
MegaRegistry_['model'].late_add(_model_loader)
MODEL = MegaRegistry_['model']
Model = MegaRegistry_['model'].add


def _criterion_loader(r: Registry):
    from catalyst.contrib.nn import criterion as m

    r.add_from_module(m)


# CRITERION = Registry("criterion")
# CRITERION.late_add(_criterion_loader)
# Criterion = CRITERION.add
MegaRegistry_['criterion'].late_add(_criterion_loader)
CRITERION = MegaRegistry_['criterion']
Criterion = MegaRegistry_['criterion'].add


def _optimizers_loader(r: Registry):
    from catalyst.contrib.nn import optimizers as m

    r.add_from_module(m)


# OPTIMIZER = Registry("optimizer")
# OPTIMIZER.late_add(_optimizers_loader)
# Optimizer = OPTIMIZER.add
MegaRegistry_['optimizer'].late_add(_optimizers_loader)
OPTIMIZER = MegaRegistry_['optimizer']
Optimizer = MegaRegistry_['optimizer'].add


def _schedulers_loader(r: Registry):
    from catalyst.contrib.nn import schedulers as m

    r.add_from_module(m)


# SCHEDULER = Registry("scheduler")
# SCHEDULER.late_add(_schedulers_loader)
# Scheduler = SCHEDULER.add
MegaRegistry_['scheduler'].late_add(_schedulers_loader)
SCHEDULER = MegaRegistry_['scheduler']
Scheduler = MegaRegistry_['scheduler'].add


def _experiments_loader(r: Registry):
    from catalyst.core.experiment import IExperiment

    r.add(IExperiment)

    from catalyst import experiments as m

    r.add_from_module(m)  # noqa: WPS347

    r.add_from_module(m)


# EXPERIMENT = Registry("experiment")
# EXPERIMENT.late_add(_experiments_loader)
# Experiment = EXPERIMENT.add
MegaRegistry_['experiment'].late_add(_experiments_loader)
EXPERIMENT = MegaRegistry_['experiment']
Experiment = MegaRegistry_['experiment'].add


def _runners_loader(r: Registry):
    from catalyst.core.runner import IRunner, IStageBasedRunner

    r.add(IRunner)
    r.add(IStageBasedRunner)

    from catalyst import runners as m  # noqa: WPS347

    r.add_from_module(m)


# RUNNER = Registry("runner")
# RUNNER.late_add(_runners_loader)
# Runner = RUNNER.add
MegaRegistry_['runner'].late_add(_runners_loader)
RUNNER = MegaRegistry_['runner']
Runner = MegaRegistry_['runner'].add


def _callbacks_loader(r: Registry):
    from catalyst.core.callback import Callback, CallbackWrapper

    r.add(Callback)
    r.add(CallbackWrapper)

    from catalyst import callbacks as m  # noqa: WPS347

    r.add_from_module(m)


# CALLBACK = Registry("callback")
# CALLBACK.late_add(_callbacks_loader)
# Callback = CALLBACK.add
MegaRegistry_['callback'].late_add(_callbacks_loader)
CALLBACK = MegaRegistry_['callback']
Callback = MegaRegistry_['callback'].add


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
