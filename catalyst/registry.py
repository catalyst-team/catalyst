import logging

from catalyst.settings import SETTINGS
from catalyst.tools import registry

logger = logging.getLogger(__name__)

REGISTRY = registry.Registry()
Registry = REGISTRY.add


def _transforms_loader(r: registry.Registry):
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
            logger.warning("kornia not available, to install kornia, " "run `pip install kornia`.")
            raise ex
    except UnsupportedNodeError as ex:
        logger.warning(
            "kornia has requirement torch>=1.5.0, probably you have"
            " an old version of torch which is incompatible.\n"
            "To update pytorch, run `pip install -U 'torch>=1.5.0'`."
        )
        if SETTINGS.kornia_required:
            raise ex


REGISTRY.late_add(_transforms_loader)


def _samplers_loader(r: registry.Registry):
    from torch.utils.data import sampler as s

    factories = {k: v for k, v in s.__dict__.items() if "Sampler" in k and k != "Sampler"}
    r.add(**factories)
    from catalyst.data import sampler

    r.add_from_module(sampler)


REGISTRY.late_add(_samplers_loader)


class _GradClipperWrap:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        self.fn(x, *self.args, **self.kwargs)


def _grad_clip_loader(r: registry.Registry):
    from torch.nn.utils import clip_grad as m

    r.add_from_module(m)


REGISTRY.late_add(_grad_clip_loader)


def _modules_loader(r: registry.Registry):
    from catalyst.contrib.nn import modules as m

    r.add_from_module(m)


REGISTRY.late_add(_modules_loader)


def _model_loader(r: registry.Registry):
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


REGISTRY.late_add(_model_loader)


def _criterion_loader(r: registry.Registry):
    from catalyst.contrib.nn import criterion as m

    r.add_from_module(m)


REGISTRY.late_add(_criterion_loader)


def _optimizers_loader(r: registry.Registry):
    from catalyst.contrib.nn import optimizers as m

    r.add_from_module(m)


REGISTRY.late_add(_optimizers_loader)


def _schedulers_loader(r: registry.Registry):
    from catalyst.contrib.nn import schedulers as m

    r.add_from_module(m)


REGISTRY.late_add(_schedulers_loader)


def _experiments_loader(r: registry.Registry):
    from catalyst.core.experiment import IExperiment

    r.add(IExperiment)

    from catalyst import experiments as m

    r.add_from_module(m)  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add(_experiments_loader)


def _runners_loader(r: registry.Registry):
    from catalyst.core.runner import IRunner, IStageBasedRunner

    r.add(IRunner)
    r.add(IStageBasedRunner)

    from catalyst import runners as m  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add(_runners_loader)


def _callbacks_loader(r: registry.Registry):
    from catalyst.core.callback import Callback, CallbackWrapper

    r.add(Callback)
    r.add(CallbackWrapper)

    from catalyst import callbacks as m  # noqa: WPS347

    r.add_from_module(m)


REGISTRY.late_add(_callbacks_loader)

__all__ = ["REGISTRY"]
