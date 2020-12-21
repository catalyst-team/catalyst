from typing import Any, Mapping, Union

import torch
import torch.nn as nn

from catalyst.core.engine import IEngine


class DeviceEngine(IEngine):
    """Single training device engine."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __repr__(self) -> str:
        return f"DeviceEngine(device='{self.device}')"

    def to_device(
        self, obj: Union[dict, torch.Tensor, nn.Module]
    ) -> Union[dict, torch.Tensor, nn.Module]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self.to_device(v)
        else:
            return obj.to(self.device)

    def handle_device(self, batch: Mapping[str, Any]):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return [self.handle_device(tensor) for tensor in batch]
        elif isinstance(batch, dict):
            return {
                key: self.handle_device(tensor)
                for key, tensor in batch.items()
            }
        return batch

    def save_checkpoint(
        self, checkpoint_content: Mapping[str, Any], file: str
    ):
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
