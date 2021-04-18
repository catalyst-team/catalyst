# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List

import torch
from torch import nn

from catalyst.contrib.models.cv.segmentation.blocks.core import DecoderBlock
from catalyst.contrib.models.cv.segmentation.blocks.unet import DecoderConcatBlock
from catalyst.contrib.models.cv.segmentation.decoder.core import DecoderSpec


class UNetDecoder(DecoderSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        block_fn: DecoderBlock = DecoderConcatBlock,
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(in_channels, in_strides)

        # features from center block
        out_channels_list = [in_channels[-1]]
        out_strides_list = [in_strides[-1]]
        # features from encoders blocks
        reversed_channels = list(reversed(in_channels[:-1]))

        blocks: List[DecoderBlock] = []
        for encoder_channels in reversed_channels:
            out_channels_list.append(encoder_channels)
            blocks.append(
                block_fn(
                    in_channels=out_channels_list[-2],
                    enc_channels=encoder_channels,
                    out_channels=out_channels_list[-1],
                    in_strides=out_strides_list[-1],
                    **kwargs
                )
            )
            out_strides_list.append(blocks[-1].out_strides)

        self.blocks = nn.ModuleList(blocks)
        self._out_channels = out_channels_list
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
        decoder_outputs = [x[-1]]
        # features from encoders blocks
        reversed_features = list(reversed(x[:-1]))

        for _i, (decoder_block, encoder_output) in enumerate(zip(self.blocks, reversed_features)):
            decoder_outputs.append(decoder_block(decoder_outputs[-1], encoder_output))

        return decoder_outputs


__all__ = ["UNetDecoder"]
