from typing import Any, Mapping, Union, Dict

import torch
import torch.nn as nn

from catalyst.core.engine import IEngine


class DeviceEngine(IEngine):
    """Single training device engine."""

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device (str, optional): use device, default is `"cpu"`.
        """
        self.device = device

    def __repr__(self) -> str:  # noqa: D105
        return f"DeviceEngine(device='{self.device}')"

    @property
    def rank(self) -> int:
        return -1

    @property
    def world_size(self) -> int:
        return 1

    def sync_device(self, tensor_or_module: Any) -> Any:
        return tensor_or_module

    def sync_tensor(self, tensor: Any) -> Any:
        return tensor

    def zero_grad(self, model, criterion, optimizer, loss) -> None:
        model.zero_grad()

    def backward_loss(self, model, criterion, optimizer, loss) -> None:
        loss.backward()

    def optimizer_step(self, model, criterion, optimizer, loss) -> None:
        optimizer.step()

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        # setup backend
        model = model_fn()
        criterion = criterion_fn()
        optimizer = optimizer_fn(model=model)
        scheduler = scheduler_fn(optimizer=optimizer)
        # @TODO: `sync_device` with the components
        return model, criterion, optimizer, scheduler

    def deinit_components(self):
        # remove backend
        pass

    # TODO: use to_device from `catalyst.utils.torch.to_device`
    def to_device(
        self, obj: Union[dict, list, tuple, torch.Tensor, nn.Module]
    ) -> Union[dict, torch.Tensor, nn.Module]:
        """Move tensors/modules to engine device.

        Args:
            obj (torch.Tensor or nn.Module or dict/list/tuple):
                object to move to device.

        Returns:
            torch.Tensor or nn.Module or dict/list/tuple where
            objects moved to a train device.
        """
        if isinstance(obj, dict):
            return {key: self.to_device(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(elem) for elem in obj)
        elif hasattr(obj, "to"):
            return obj.to(self.device)
        else:
            return obj

    # TODO: think about using `self.to_device`
    def handle_device(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Move batch to a device.

        Args:
            batch (Mapping[str, Any]): data which should be moved
                to a device.

        Returns:
            Mapping[str, Any] where each torch.Tensor object will
            be on a training device.
        """
        return self.to_device(batch)

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
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
        model = checkpoint["model"]
        criterion = checkpoint["criterion"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]

    def save_checkpoint(self, checkpoint_content: Mapping[str, Any], file: str):
        torch.save(checkpoint_content, file)

    def load_checkpoint(
        self,
        file: str,
        model: Union[nn.Module, nn.DataParallel],
        optimizer: nn.Module = None,
        criterion=None,
        scheduler=None,
    ):
        content = torch.load(file)

        if "model_state_dict" in content:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(content["model_state_dict"])
            elif isinstance(model, nn.Module):
                model.load_state_dict(content["model_state_dict"])

        if "optimizer_state_dict" in content and optimizer is not None:
            optimizer.load_state_dict(content["optimizer_state_dict"])

        if "criterion_state_dict" in content and criterion is not None:
            criterion.load_state_dict(content["criterion_state_dict"])

        if "scheduler_state_dict" in content and scheduler is not None:
            scheduler.load_state_dict(content["scheduler_state_dict"])
