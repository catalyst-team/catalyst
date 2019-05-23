from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import HeadSpec
from ..blocks import UnetUpsampleBlock


class BaseUnetHead(HeadSpec):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_upsample_blocks: int = 0,
        upsample_scale: int = None,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        super().__init__()
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        upsamples = (
                [UnetUpsampleBlock(in_channels, in_channels)]
                * num_upsample_blocks)
        self.head = nn.Sequential(
            nn.Dropout2d(p=dropout, inplace=True),
            *upsamples,
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ = x[-1]
        x = self.head(x_)
        if self.upsample_scale is not None:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        return x
