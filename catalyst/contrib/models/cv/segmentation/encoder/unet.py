# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List

import torch
from torch import nn

from catalyst.contrib.models.cv.segmentation.blocks.unet import EncoderDownsampleBlock
from catalyst.contrib.models.cv.segmentation.encoder.core import (  # noqa: WPS450, E501
    _take,
    EncoderSpec,
)


class UnetEncoder(EncoderSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_blocks: int,
        layers_indices: List[int] = None,
        **kwargs,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        self.num_filters = num_channels
        self.num_blocks = num_blocks
        self._layers_indices = layers_indices or list(range(num_blocks))

        self._channels = [self.num_filters * 2 ** i for i in range(self.num_blocks)]
        self._strides = [2 ** (i) for i in range(self.num_blocks)]
        self._channels = _take(self._channels, self._layers_indices)
        self._strides = _take(self._strides, self._layers_indices)

        for i in range(num_blocks):
            in_channels = in_channels if not i else num_channels * 2 ** (i - 1)
            out_channels = num_channels * 2 ** i
            self.add_module(
                f"block{i + 1}",
                EncoderDownsampleBlock(in_channels, out_channels, first_stride=1, **kwargs),
            )
            if i != self.num_blocks - 1:
                self.add_module(f"pool{i + 1}", nn.MaxPool2d(2, 2))

    @property
    def out_channels(self) -> List[int]:
        """Number of channels produced by the block."""
        return self._channels

    @property
    def out_strides(self) -> List[int]:
        """@TODO: Docs. Contribution is welcome."""
        return self._strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward call."""
        output = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i + 1}")(x)
            output.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f"pool{i + 1}")(x)
        output = _take(output, self._layers_indices)
        return output


__all__ = ["UnetEncoder"]
