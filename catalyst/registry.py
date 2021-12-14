import logging

from catalyst.settings import SETTINGS
import hydra_slayer

logger = logging.getLogger(__name__)

REGISTRY = hydra_slayer.Registry()
Registry = REGISTRY.add


def _data_loader(r: hydra_slayer.Registry):
    from catalyst import data as m

    r.add_from_module(m)

    if SETTINGS.albu_required:
        import albumentations as m

        r.add_from_module(m, prefix=["A.", "albu.", "albumentations."])

        from albumentations import pytorch as p

        r.add_from_module(p, prefix=["A.", "albu.", "albumentations."])


REGISTRY.late_add(_data_loader)


def _datasets_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import datasets

    r.add_from_module(datasets)


REGISTRY.late_add(_datasets_loader)


def _samplers_loader(r: hydra_slayer.Registry):
    from torch.utils.data import sampler as s

    factories = {k: v for k, v in s.__dict__.items() if "Sampler" in k and k != "Sampler"}
    r.add(**factories)


REGISTRY.late_add(_samplers_loader)


def _dataloaders_loader(r: hydra_slayer.Registry):
    from torch.utils.data import DataLoader

    r.add(DataLoader)


REGISTRY.late_add(_dataloaders_loader)


def _grad_clip_loader(r: hydra_slayer.Registry):
    from torch.nn.utils import clip_grad as m

    r.add_from_module(m)


REGISTRY.late_add(_grad_clip_loader)


def _modules_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import layers as m

    r.add_from_module(m)


REGISTRY.late_add(_modules_loader)


def _model_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import models as m

    r.add_from_module(m)


REGISTRY.late_add(_model_loader)


def _criterion_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import losses as m

    r.add_from_module(m)


REGISTRY.late_add(_criterion_loader)


def _optimizers_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import optimizers as m

    r.add_from_module(m)

    if SETTINGS.fairscale_required:
        from fairscale import optim as m2

        r.add_from_module(m2, prefix=["fairscale."])


REGISTRY.late_add(_optimizers_loader)


def _schedulers_loader(r: hydra_slayer.Registry):
    from catalyst.contrib import schedulers as m

    r.add_from_module(m)


REGISTRY.late_add(_schedulers_loader)


def _engines_loader(r: hydra_slayer.Registry):
    from catalyst.core.engine import IEngine

    r.add(IEngine)

    from catalyst import engines as m

    r.add_from_module(m)


REGISTRY.late_add(_engines_loader)


def _runners_loader(r: hydra_slayer.Registry):
    from catalyst.core import runner

    r.add_from_module(runner)

    from catalyst import runners as m

    r.add_from_module(m)


REGISTRY.late_add(_runners_loader)


def _callbacks_loader(r: hydra_slayer.Registry):
    from catalyst.core import callback

    r.add_from_module(callback)

    from catalyst import callbacks as m

    r.add_from_module(m)


REGISTRY.late_add(_callbacks_loader)


def _loggers_loader(r: hydra_slayer.Registry):
    from catalyst.core import logger

    r.add_from_module(logger)

    from catalyst import loggers as m

    r.add_from_module(m)


REGISTRY.late_add(_loggers_loader)


def _torch_functional_loader(r: hydra_slayer.Registry):
    import torch.nn.functional as F

    r.add_from_module(F, ["F."])


REGISTRY.late_add(_torch_functional_loader)


def _torch_loader(r: hydra_slayer.Registry):
    import torch as m

    r.add_from_module(m, ["torch."], ignore_all=True)


REGISTRY.late_add(_torch_loader)


__all__ = ["REGISTRY"]
