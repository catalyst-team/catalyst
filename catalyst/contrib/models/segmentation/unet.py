from typing import Dict
from functools import partial
import numpy as np

from .blocks import EncoderDownsampleBlock, EncoderUpsampleBlock, \
    DecoderConcatBlock
from .encoder import UnetEncoder, ResnetEncoder
from .bridge import UnetBridge
from .decoder import UNetDecoder
from .head import UnetHead
from .core import UnetSpec, ResnetUnetSpec


class Unet(UnetSpec):
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
            block_fn=DecoderConcatBlock,
            **decoder_params,
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            num_upsample_blocks=int(np.log2(decoder.out_strides[-1])),
            **head_params
        )

        return encoder, bridge, decoder, head


class ResnetUnet(ResnetUnetSpec):
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
            **bridge_params
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            in_strides=bridge.out_strides,
            block_fn=partial(
                DecoderConcatBlock, aggregate_first=True, upsample_scale=2
            ),
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
