from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import HeadSpec
from ..blocks import EncoderUpsampleBlock


class UnetHead(HeadSpec):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        in_strides: List[int] = None,
        dropout: float = 0.0,
        num_upsample_blocks: int = 0,
        upsample_scale: int = 1,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        super().__init__(in_channels, out_channels, in_strides)
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        in_channels_ = in_channels[-1]
        additional_layers = [
            EncoderUpsampleBlock(in_channels_, in_channels_)
        ] * num_upsample_blocks
        if dropout > 0:
            additional_layers.append(nn.Dropout2d(p=dropout, inplace=True))
        self.head = nn.Sequential(
            *additional_layers,
            nn.Conv2d(in_channels_, out_channels, 1)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ = x[-1]
        x = self.head(x_)
        if self.upsample_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        return x
