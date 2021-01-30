from typing import Any, Dict, Mapping, Union
import functools

import torch
import torch.nn as nn

# TODO: use only latest version of pytorch
import torch.cuda.amp as amp

from catalyst.engines.device import DeviceEngine


def _patch_with_scaler(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with amp.autocast():
            res = fn(*args, **kwargs)
        return res

    return wrapper


# TODO: wrap with autocast computing output & loss
class AMPEngine(DeviceEngine):
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device (str, optional): use device, default is `"cpu"`.
        """
        super().__init__(device)
        self.scaler = amp.GradScaler()

    def __repr__(self) -> str:  # noqa: D105
        return f"AMPEngine(device='{self.device}')"

    def backward_loss(self, model, criterion, optimizer, loss) -> None:
        self.scaler.scale(loss).backward()

    def optimizer_step(self, model, criterion, optimizer, loss) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        # @TODO: how could we do better?)
        # model
        model = model_fn()
        model = self.sync_device(model)
        model.forward = amp.autocast()(model.forward)
        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        criterion.__call__ = amp.autocast()(criterion.__call__)
        # optimizer
        optimizer = optimizer_fn(model=model)
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn(optimizer=optimizer)
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler
