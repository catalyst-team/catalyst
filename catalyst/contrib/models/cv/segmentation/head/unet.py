# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from catalyst.contrib.models.cv.segmentation.blocks import EncoderUpsampleBlock
from catalyst.contrib.models.cv.segmentation.head.core import HeadSpec


class UnetHead(HeadSpec):
    """@TODO: Docs. Contribution is welcome."""

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
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(in_channels, out_channels, in_strides)
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        in_channels_last = in_channels[-1]
        additional_layers = [
            EncoderUpsampleBlock(in_channels_last, in_channels_last)
        ] * num_upsample_blocks
        if dropout > 0:
            additional_layers.append(nn.Dropout2d(p=dropout, inplace=True))
        self.head = nn.Sequential(*additional_layers, nn.Conv2d(in_channels_last, out_channels, 1))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward call."""
        x_last = x[-1]
        x = self.head(x_last)
        if self.upsample_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
        return x


__all__ = ["UnetHead"]
