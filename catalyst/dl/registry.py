import logging
import os

from catalyst.contrib.registry import (
    Criterion, CRITERIONS, GRAD_CLIPPERS, Model, MODELS, Module, MODULES,
    Optimizer, OPTIMIZERS, Sampler, SAMPLERS, Scheduler, SCHEDULERS
)
from ..utils.registry import Registry

logger = logging.getLogger(__name__)


def _callbacks_loader(r: Registry):
    from catalyst.dl import callbacks as m
    r.add_from_module(m)


CALLBACKS = Registry("callback")
CALLBACKS.late_add(_callbacks_loader)
Callback = CALLBACKS.add


def _transforms_loader(r: Registry):
    try:
        import albumentations as m
        r.add_from_module(m)

        from albumentations import pytorch as p
        r.add_from_module(p)

        from catalyst.contrib import transforms as t
        r.add_from_module(t, prefix="catalsyt.")
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
