from typing import Any, Dict, Mapping, Union

import torch
import torch.nn as nn

from catalyst.engines.device import DeviceEngine


class DataParallelEngine(DeviceEngine):
    def __init__(self):
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"DataParallelEngine(device_count={self.device_count})"

    # def to_device(
    #     self, obj: Union[dict, torch.Tensor, nn.Module]
    # ) -> Union[dict, torch.Tensor, nn.Module]:
    #     # fmt: off
    #     if isinstance(obj, dict):
    #         for k, v in obj.items():
    #             obj[k] = self.to_device(v)
    #     elif isinstance(obj, nn.Module) \
    #         and not isinstance(obj, nn.DataParallel):
    #         return nn.DataParallel(obj)
    #     else:
    #         return obj.to(self.device)
    #     # fmt: on

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        # unwrap model
        _model = model.module if isinstance(model, nn.DataParallel) else model
        return super().pack_checkpoint(_model, criterion, optimizer, scheduler, **kwargs)

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        # TODO: method for unpacking torch.nn.DataParallel
        torch.save(checkpoint, path)
