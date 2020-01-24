import logging
import os

from catalyst.contrib.registry import (
    Criterion, CRITERIONS, GRAD_CLIPPERS, Model, MODELS, Module, MODULES,
    Optimizer, OPTIMIZERS, Sampler, SAMPLERS, Scheduler, SCHEDULERS
)
from catalyst.utils.registry import Registry
from catalyst.core.registry import CALLBACKS, Callback

logger = logging.getLogger(__name__)


def _callbacks_loader(r: Registry):
    from catalyst.dl import callbacks as m
    r.add_from_module(m)


CALLBACKS.late_add(_callbacks_loader)


def _transforms_loader(r: Registry):
    try:
        import albumentations as m
        r.add_from_module(m)
    except ImportError as ex:
        logger.warning(
            "albumentations not available, to install albumentations, "
            "run `pip install albumentations`."
        )
        if os.environ.get("USE_ALBUMENTATIONS", "0") == "1":
            raise ex


TRANSFORMS = Registry("transform")
TRANSFORMS.late_add(_transforms_loader)
Transform = TRANSFORMS.add

__all__ = [
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Callback",
    "Module",
    "Model",
    "Sampler",
    "Transform",
    "MODULES",
    "CRITERIONS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "GRAD_CLIPPERS",
    "MODELS",
    "SAMPLERS",
    "TRANSFORMS",
]
