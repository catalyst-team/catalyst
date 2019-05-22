from typing import List

import torch
import torch.nn as nn

from .core import EncoderSpec
from ..unet_blocks import UnetEncoderBlock


class UnetEncoder(EncoderSpec):
    def __init__(self, in_channels: int, num_channels: int, num_blocks: int):
        super().__init__()

        self.num_filters = num_channels
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            in_channels = in_channels if not i else num_channels * 2 ** (i - 1)
            out_channels = num_channels * 2 ** i
            self.add_module(
                f"block{i + 1}", UnetEncoderBlock(in_channels, out_channels)
            )
            self.add_module(f"pool{i + 1}", nn.MaxPool2d(2, 2))

    @property
    def layers(self) -> List[int]:
        return list(range(self.num_blocks))

    @property
    def out_channels(self) -> List[int]:
        return [self.num_filters * 2**i for i in range(self.num_blocks)]

    @property
    def out_strides(self) -> List[int]:
        return [1] * self.num_blocks

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [
            self.__getattr__(f"block{i + 1}")
            for i in range(self.num_blocks)
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i + 1}")(x)
            acts.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f"pool{i + 1}")(x)
        return acts
