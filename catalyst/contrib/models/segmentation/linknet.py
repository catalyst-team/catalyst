from typing import Dict
from functools import partial

from .blocks import EncoderDownsampleBlock, DecoderSumBlock

from .encoder import UnetEncoder, ResnetEncoder
from .bridge import UnetBridge
from .decoder import UNetDecoder
from .head import UnetHead
from .core import _UnetSpec, _ResnetUnetSpec


class Linknet(_UnetSpec):

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
            **head_params
        )
        return encoder, bridge, decoder, head


class ResnetLinknet(_ResnetUnetSpec):

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
            block_fn=partial(
                DecoderSumBlock,
                aggregate_first=False,
                upsample_scale=None),
            **decoder_params
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=6 - len(encoder.out_strides),
            **head_params
        )
        return encoder, bridge, decoder, head
