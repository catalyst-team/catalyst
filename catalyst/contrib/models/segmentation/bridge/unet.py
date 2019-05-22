from typing import List
import torch

from .core import BridgeSpec
from ..unet_blocks import UnetCentralBlock


class UnetBridge(BridgeSpec):
    def __init__(self, in_channels: List[int], out_channels: int, **kwargs):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = in_channels + [out_channels]
        self.block = UnetCentralBlock(
            in_channels=in_channels[-1],
            out_channels=out_channels,
            **kwargs
        )

    @property
    def in_channels(self) -> List[int]:
        return self._in_channels

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x_: torch.Tensor = x[-1]
        x_: torch.Tensor = self.block(x_)
        output = x + [x_]
        return output
