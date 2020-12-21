from typing import Any, Mapping, Union

import torch
import torch.nn as nn

from catalyst.engines.device import DeviceEngine


class DataParallelEngine(DeviceEngine):
    def __init__(self, device: str = "cuda:0"):
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:
        return f"DataParallelDeviceEngine(device_count={self.device_count})"

    def to_device(
        self, obj: Union[dict, torch.Tensor, nn.Module]
    ) -> Union[dict, torch.Tensor, nn.Module]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self.to_device(v)
        elif isinstance(obj, nn.Module):
            return nn.DataParallel(obj)
        else:
            return obj.to(self.device)

    def save_checkpoint(
        self, checkpoint_content: Mapping[str, Any], file: str
    ):
        # TODO: method for unpacking torch.nn.DataParallel
        torch.save(checkpoint_content, file)
