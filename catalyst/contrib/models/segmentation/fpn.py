from typing import Dict

from .blocks import EncoderDownsampleBlock
from .encoder import UnetEncoder, ResnetEncoder
from .bridge import UnetBridge
from .decoder import FPNDecoder
from .head import FPNHead
from .core import UnetSpec, ResnetUnetSpec


class FPNUnet(UnetSpec):

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
        decoder = FPNDecoder(
            in_channels=bridge.out_channels,
            in_strides=bridge.out_strides,
            **decoder_params
        )
        head = FPNHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            upsample_scale=decoder.out_strides[-1],
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head


class ResnetFPNUnet(ResnetUnetSpec):

    def _get_components(
        self,
        encoder: ResnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = None
        decoder = FPNDecoder(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            **decoder_params
        )
        head = FPNHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            upsample_scale=decoder.out_strides[-1],
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head
