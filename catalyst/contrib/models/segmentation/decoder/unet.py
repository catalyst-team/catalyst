from typing import List

import torch
import torch.nn as nn

from .core import DecoderSpec
from ..unet_blocks import UnetDecoderBlock


class UNetDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        dilation_factors: List[int] = None,
    ):
        super().__init__()

        # features from center block
        out_channels = [in_channels[-1]]
        # features from encoder blocks
        reversed_features = list(reversed(in_channels[:-1]))

        if dilation_factors is None:
            dilation_factors = [1] * len(reversed_features)

        blocks = []
        for block_index, encoder_features in enumerate(reversed_features):
            blocks.append(
                UnetDecoderBlock(
                    in_channels=out_channels[-1] + encoder_features,
                    out_channels=encoder_features,
                    dilation=dilation_factors[block_index]))
            out_channels.append(encoder_features)

        self.blocks = nn.ModuleList(blocks)
        self._out_channels = out_channels

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: List[torch.Tensor]):
        # features from center block
        decoder_outputs = [x[-1]]
        # features from encoder blocks
        reversed_features = list(reversed(x[:-1]))

        for i, (decoder_block, encoder_output) \
                in enumerate(zip(self.blocks, reversed_features)):
            decoder_outputs.append(
                decoder_block(decoder_outputs[-1], encoder_output))

        return decoder_outputs
