from typing import List
import torch

from .core import BridgeSpec
from ..blocks import EncoderBlock, EncoderDownsampleBlock


class UnetBridge(BridgeSpec):
    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        out_channels: int,
        block_fn: EncoderBlock = EncoderDownsampleBlock,
        **kwargs
    ):
        super().__init__(in_channels, in_strides)

        self.block = block_fn(
            in_channels=in_channels[-1],
            in_strides=in_strides[-1],
            out_channels=out_channels,
            **kwargs
        )

        self._out_channels = in_channels + [self.block.out_channels]
        self._out_strides = in_strides + [self.block.out_strides]

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    @property
    def out_strides(self) -> List[int]:
        return self._out_strides

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x_: torch.Tensor = x[-1]
        x_: torch.Tensor = self.block(x_)
        output = x + [x_]
        return output
