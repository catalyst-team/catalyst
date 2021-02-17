from typing import Dict
from functools import partial

import numpy as np

from catalyst.contrib.models.cv.segmentation.blocks import (
    DecoderConcatBlock,
    EncoderDownsampleBlock,
    EncoderUpsampleBlock,
)
from catalyst.contrib.models.cv.segmentation.bridge import UnetBridge
from catalyst.contrib.models.cv.segmentation.core import ResnetUnetSpec, UnetSpec
from catalyst.contrib.models.cv.segmentation.decoder import UNetDecoder
from catalyst.contrib.models.cv.segmentation.encoder import ResnetEncoder, UnetEncoder
from catalyst.contrib.models.cv.segmentation.head import UnetHead


class Unet(UnetSpec):
    """@TODO: Docs. Contribution is welcome."""

    def _get_components(
        self,
        encoder: UnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = UnetBridge(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            out_channels=encoder.out_channels[-1] * 2,
            block_fn=EncoderDownsampleBlock,
            **bridge_params,
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            in_strides=bridge.out_strides,
            block_fn=DecoderConcatBlock,
            **decoder_params,
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=int(np.log2(decoder.out_strides[-1])),
            **head_params,
        )

        return encoder, bridge, decoder, head


class ResnetUnet(ResnetUnetSpec):
    """@TODO: Docs. Contribution is welcome."""

    def _get_components(
        self,
        encoder: ResnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = UnetBridge(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            out_channels=encoder.out_channels[-1],
            block_fn=partial(EncoderUpsampleBlock, pool_first=True),
            **bridge_params,
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            in_strides=bridge.out_strides,
            block_fn=partial(DecoderConcatBlock, aggregate_first=True, upsample_scale=2),
            **decoder_params,
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=int(np.log2(decoder.out_strides[-1])),
            **head_params,
        )
        return encoder, bridge, decoder, head


__all__ = ["Unet", "ResnetUnet"]
