from typing import List

import torch
import torch.nn as nn

from .core import DecoderSpec
from ..blocks.fpn import DecoderFPNBlock


class FPNDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        pyramid_channels: int = 256,
        **kwargs
    ):
        super().__init__(in_channels, in_strides)
        out_strides_ = [in_strides[-1]]

        self.center_conv = nn.Conv2d(
            in_channels[-1],
            pyramid_channels,
            kernel_size=1)

        # features from encoder blocks
        reversed_features = list(reversed(in_channels[:-1]))

        blocks = []
        for encoder_features in reversed_features:
            blocks.append(
                DecoderFPNBlock(
                    in_channels=pyramid_channels,
                    enc_channels=encoder_features,
                    out_channels=pyramid_channels,
                    in_strides=out_strides_[-1],
                    **kwargs
                ))
            out_strides_.append(blocks[-1].out_strides)
        self.blocks = nn.ModuleList(blocks)
        self._out_channels = [pyramid_channels] * len(in_channels)
        self._out_strides = out_strides_

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    @property
    def out_strides(self) -> List[int]:
        return self._out_strides

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # features from center block
        fpn_features = [self.center_conv(x[-1])]
        # features from encoder blocks
        reversed_features = list(reversed(x[:-1]))

        for i, (fpn_block, encoder_output) \
                in enumerate(zip(self.blocks, reversed_features)):
            fpn_features.append(
                fpn_block(fpn_features[-1], encoder_output))

        return fpn_features
