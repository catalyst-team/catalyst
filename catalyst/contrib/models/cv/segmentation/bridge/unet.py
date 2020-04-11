from typing import List

import torch

from ..blocks import EncoderBlock, EncoderDownsampleBlock
from .core import BridgeSpec


class UnetBridge(BridgeSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        out_channels: int,
        block_fn: EncoderBlock = EncoderDownsampleBlock,
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
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
        """Number of channels produced by the block."""
        return self._out_channels

    @property
    def out_strides(self) -> List[int]:
        """@TODO: Docs. Contribution is welcome."""
        return self._out_strides

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward call."""
        x_: torch.Tensor = x[-1]
        x_: torch.Tensor = self.block(x_)
        output = x + [x_]
        return output
