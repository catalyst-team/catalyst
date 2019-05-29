from typing import List

import torch
import torch.nn as nn

from .core import DecoderSpec
from ..blocks.core import DecoderBlock
from ..blocks.unet import DecoderConcatBlock


class UNetDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        block_fn: DecoderBlock = DecoderConcatBlock,
        **kwargs
    ):
        super().__init__(in_channels, in_strides)

        # features from center block
        out_channels_ = [in_channels[-1]]
        out_strides_ = [in_strides[-1]]
        # features from encoder blocks
        reversed_channels = list(reversed(in_channels[:-1]))

        blocks: List[DecoderBlock] = []
        for encoder_channels in reversed_channels:
            out_channels_.append(encoder_channels)
            blocks.append(
                block_fn(
                    in_channels=out_channels_[-2],
                    enc_channels=encoder_channels,
                    out_channels=out_channels_[-1],
                    in_strides=out_strides_[-1],
                    **kwargs))
            out_strides_.append(blocks[-1].out_strides)

        self.blocks = nn.ModuleList(blocks)
        self._out_channels = out_channels_
        self._out_strides = out_strides_

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    @property
    def out_strides(self) -> List[int]:
        return self._out_strides

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
