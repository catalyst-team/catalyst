import logging

from catalyst.registry.registry import Registry
from catalyst.registry.subregistry import SubRegistry
from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

REGISTRY = Registry()


def _transforms_loader(r: SubRegistry):
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


REGISTRY.late_add("transform", _transforms_loader)


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


REGISTRY.late_add("sampler", _samplers_loader)


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
REGISTRY.add_subregistry("func", default_meta_factory=_GradClipperWrap)
REGISTRY.late_add("func", _grad_clip_loader)


def _modules_loader(r: Registry):
    from catalyst.contrib.nn import modules as m

    r.add_from_module(m)


REGISTRY.late_add("module", _modules_loader)


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


REGISTRY.late_add("model", _model_loader)


def _criterion_loader(r: Registry):
    from catalyst.contrib.nn import criterion as m

    r.add_from_module(m)


REGISTRY.late_add("criterion", _criterion_loader)


def _optimizers_loader(r: Registry):
    from catalyst.contrib.nn import optimizers as m

    r.add_from_module(m)


REGISTRY.late_add("optimizer", _optimizers_loader)


def _schedulers_loader(r: Registry):
    from catalyst.contrib.nn import schedulers as m

    r.add_from_module(m)


REGISTRY.late_add("scheduler", _schedulers_loader)


def _experiments_loader(r: Registry):
    from catalyst.core.experiment import IExperiment

    r.add(IExperiment)

    from catalyst import experiments as m

    r.add_from_module(m)  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add("experiment", _experiments_loader)


def _runners_loader(r: Registry):
    from catalyst.core.runner import IRunner, IStageBasedRunner

    r.add(IRunner)
    r.add(IStageBasedRunner)

    from catalyst import runners as m  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add("runner", _runners_loader)


def _callbacks_loader(r: Registry):
    from catalyst.core.callback import Callback, CallbackWrapper

    r.add(Callback)
    r.add(CallbackWrapper)

    from catalyst import callbacks as m  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add("callback", _callbacks_loader)


# backward compatibility
CALLBACKS = CALLBACK = REGISTRY["callback"]
Callback = REGISTRY["callback"].add
CRITERIONS = CRITERION = REGISTRY["criterion"]
Criterion = REGISTRY["criterion"].add
EXPERIMENTS = EXPERIMENT = REGISTRY["experiment"]
Experiment = REGISTRY["experiment"].add
GRAD_CLIPPERS = GRAD_CLIPPER = REGISTRY["func"]
MODELS = MODEL = REGISTRY["model"]
Model = REGISTRY["model"].add
MODULES = MODULE = REGISTRY["module"]
Module = REGISTRY["module"].add
OPTIMIZERS = OPTIMIZER = REGISTRY["optimizer"]
Optimizer = REGISTRY["optimizer"].add
RUNNERS = RUNNER = REGISTRY["runner"]
Runner = REGISTRY["runner"].add
SAMPLERS = SAMPLER = REGISTRY["sampler"]
Sampler = REGISTRY["sampler"].add
SCHEDULERS = SCHEDULER = REGISTRY["scheduler"]
Scheduler = REGISTRY["scheduler"].add
TRANSFORMS = TRANSFORM = REGISTRY["transform"]
Transform = REGISTRY["transform"].add


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
