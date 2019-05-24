from typing import List

import torch
import torch.nn as nn

from .core import DecoderSpec
from ..blocks.fpn import FPNBlock, SegmentationBlock


class FPNDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        **kwargs
    ):
        super().__init__()
        self._out_channels = [segmentation_channels]

        self.conv1 = nn.Conv2d(
            in_channels[-1],
            pyramid_channels,
            kernel_size=1)

        # features from encoder blocks
        reversed_features = list(reversed(in_channels[:-1]))

        fpn_block = []
        for encoder_features in reversed_features:
            fpn_block.append(FPNBlock(pyramid_channels, encoder_features))
        self.fpn_block = nn.ModuleList(fpn_block)

        segmentation_blocks = []
        for i in range(len(in_channels)):
            segmentation_blocks.append(
                SegmentationBlock(
                    pyramid_channels,
                    segmentation_channels,
                    num_upsamples=i))
        segmentation_blocks = list(reversed(segmentation_blocks))
        self.segmentation_blocks = nn.ModuleList(segmentation_blocks)

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # features from center block
        fpn_features = [self.conv1(x[-1])]
        # features from encoder blocks
        reversed_features = list(reversed(x[:-1]))

        for i, (fpn_block, encoder_output) \
                in enumerate(zip(self.fpn_block, reversed_features)):
            fpn_features.append(
                fpn_block(fpn_features[-1], encoder_output))

        segmentation_features = list(map(
            lambda block, features: block(features),
            self.segmentation_blocks,
            fpn_features))

        x = sum(segmentation_features)

        return [x]
