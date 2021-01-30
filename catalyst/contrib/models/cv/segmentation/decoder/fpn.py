# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List

import torch
from torch import nn

from catalyst.contrib.models.cv.segmentation.blocks.fpn import DecoderFPNBlock
from catalyst.contrib.models.cv.segmentation.decoder.core import DecoderSpec


class FPNDecoder(DecoderSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self, in_channels: List[int], in_strides: List[int], pyramid_channels: int = 256, **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(in_channels, in_strides)
        out_strides_list = [in_strides[-1]]

        self.center_conv = nn.Conv2d(in_channels[-1], pyramid_channels, kernel_size=1)

        # features from encoders blocks
        reversed_features = list(reversed(in_channels[:-1]))

        blocks = []
        for encoder_features in reversed_features:
            blocks.append(
                DecoderFPNBlock(
                    in_channels=pyramid_channels,
                    enc_channels=encoder_features,
                    out_channels=pyramid_channels,
                    in_strides=out_strides_list[-1],
                    **kwargs
                )
            )
            out_strides_list.append(blocks[-1].out_strides)
        self.blocks = nn.ModuleList(blocks)
        self._out_channels = [pyramid_channels] * len(in_channels)
        self._out_strides = out_strides_list

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
        # features from center block
        fpn_features = [self.center_conv(x[-1])]
        # features from encoders blocks
        reversed_features = list(reversed(x[:-1]))

        for _i, (fpn_block, encoder_output) in enumerate(zip(self.blocks, reversed_features)):
            fpn_features.append(fpn_block(fpn_features[-1], encoder_output))

        return fpn_features


__all__ = ["FPNDecoder"]
