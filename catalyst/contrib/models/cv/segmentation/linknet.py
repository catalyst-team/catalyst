from typing import Dict
from functools import partial

import numpy as np

from catalyst.contrib.models.cv.segmentation.blocks import DecoderSumBlock, EncoderDownsampleBlock
from catalyst.contrib.models.cv.segmentation.bridge import UnetBridge
from catalyst.contrib.models.cv.segmentation.core import ResnetUnetSpec, UnetSpec
from catalyst.contrib.models.cv.segmentation.decoder import UNetDecoder
from catalyst.contrib.models.cv.segmentation.encoder import ResnetEncoder, UnetEncoder
from catalyst.contrib.models.cv.segmentation.head import UnetHead


class Linknet(UnetSpec):
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
            **bridge_params
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            in_strides=bridge.out_strides,
            block_fn=DecoderSumBlock,
            **decoder_params
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=int(np.log2(decoder.out_strides[-1])),
            **head_params
        )
        return encoder, bridge, decoder, head


class ResnetLinknet(ResnetUnetSpec):
    """@TODO: Docs. Contribution is welcome."""

    def _get_components(
        self,
        encoder: ResnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = None
        decoder = UNetDecoder(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            block_fn=partial(DecoderSumBlock, aggregate_first=False, upsample_scale=None),
            **decoder_params
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=int(np.log2(decoder.out_strides[-1])),
            **head_params
        )
        return encoder, bridge, decoder, head


__all__ = ["Linknet", "ResnetLinknet"]
