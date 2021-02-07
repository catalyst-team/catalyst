from typing import Any, Dict, Mapping, Union
from contextlib import contextmanager

import torch

# TODO: works only with latest pytorch (1.7.1) - fix for older versions
import torch.cuda.amp as amp
import torch.nn as nn

from catalyst.engines.device import DeviceEngine


class AMPEngine(DeviceEngine):
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device (str): use device, default is `"cpu"`.
        """
        super().__init__(device)
        self.scaler = amp.GradScaler()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def backward_loss(self, loss, model, optimizer) -> None:
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        # TODO: how could we do better?)
        # model
        model = model_fn()
        model = self.sync_device(model)
        # model.forward = amp.autocast()(model.forward)
        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # criterion.__call__ = amp.autocast()(criterion.__call__)
        # optimizer
        optimizer = optimizer_fn(model=model)
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn(optimizer=optimizer)
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        return amp.autocast()
