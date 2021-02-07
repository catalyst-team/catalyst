from typing import Any, Dict, Mapping, Union

import numpy as np
import torch
import torch.nn as nn

from catalyst.core.engine import IEngine


# @TODO: merge it with DataParallel version?
class DeviceEngine(IEngine):
    """Single training device engine."""

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device (str, optional): use device, default is `"cpu"`.
        """
        self.device = device

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    @property
    def rank(self) -> int:
        return -1

    @property
    def world_size(self) -> int:
        return 1

    def sync_device(
        self, tensor_or_module: Union[dict, list, tuple, torch.Tensor, nn.Module]
    ) -> Any:
        if isinstance(tensor_or_module, dict):
            return {key: self.sync_device(value) for key, value in tensor_or_module.items()}
        elif isinstance(tensor_or_module, (list, tuple)):
            return type(tensor_or_module)(self.sync_device(elem) for elem in tensor_or_module)
        elif torch.is_tensor(tensor_or_module):
            return tensor_or_module.to(self.device, non_blocking=True)
        elif (
            isinstance(tensor_or_module, (np.ndarray, np.void))
            and tensor_or_module.dtype.fields is not None
        ):
            return {
                k: self.sync_device(tensor_or_module[k])
                for k in tensor_or_module.dtype.fields.keys()
            }
        elif isinstance(tensor_or_module, np.ndarray):
            return torch.tensor(tensor_or_module, device=self.device)
        elif isinstance(tensor_or_module, nn.Module):
            return tensor_or_module.to(self.device)
        # elif hasattr(tensor_or_module, "to"):
        #     return tensor_or_module.to(self.device)
        return tensor_or_module

    def sync_tensor(self, tensor: Any) -> Any:
        return tensor

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        # @TODO: how could we do better?)
        # model
        model = model_fn()
        model = self.sync_device(model)
        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # optimizer
        optimizer = optimizer_fn(model=model)
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn(optimizer=optimizer)
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def deinit_components(self):
        # remove backend
        pass

    def zero_grad(self, loss, model, optimizer) -> None:
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        optimizer.step()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        # add data parallel support
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            **kwargs,
        }

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:

        if "model_state_dict" in checkpoint:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(model, nn.Module):
                model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "criterion_state_dict" in checkpoint and criterion is not None:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])

        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        return torch.load(path)
