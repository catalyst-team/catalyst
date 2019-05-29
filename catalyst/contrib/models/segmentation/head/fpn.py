from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import HeadSpec
from ..blocks import EncoderUpsampleBlock, SegmentationBlock


class FPNHead(HeadSpec):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        hid_channel: int = 256,
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

        segmentation_blocks = []
        for i, in_channels_ in enumerate(in_channels):
            if in_strides is not None:
                i = np.log2(in_strides[i]) \
                    - num_upsample_blocks - np.log2(upsample_scale)
            segmentation_blocks.append(
                SegmentationBlock(
                    in_channels=in_channels_,
                    out_channels=hid_channel,
                    num_upsamples=int(i)))
        self.segmentation_blocks = nn.ModuleList(segmentation_blocks)

        additional_layers = [
            EncoderUpsampleBlock(hid_channel, hid_channel)
        ] * num_upsample_blocks
        if dropout > 0:
            additional_layers.append(nn.Dropout2d(p=dropout, inplace=True))
        self.head = nn.Sequential(
            *additional_layers,
            nn.Conv2d(hid_channel, out_channels, 1)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = list(map(
            lambda block, features: block(features),
            self.segmentation_blocks,
            x))
        x = sum(x)
        x = self.head(x)
        if self.upsample_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        return x
