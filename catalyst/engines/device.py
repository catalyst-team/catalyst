from typing import Any, Mapping, Union

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
