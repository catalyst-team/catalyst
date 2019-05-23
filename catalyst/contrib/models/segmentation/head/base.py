from typing import List

import torch
import torch.nn as nn

from .core import HeadSpec
from ..blocks import UnetUpsampleBlock


class BaseUnetHead(HeadSpec):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_first: bool = False
    ):
        super().__init__()
        if upsample_first:
            self.head = nn.Sequential(
                UnetUpsampleBlock(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        else:
            self.head = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ = x[-1]
        output = self.head(x_)
        return output
