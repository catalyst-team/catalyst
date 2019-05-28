from typing import Dict

from .encoder import UnetEncoder, ResnetEncoder
from .decoder import PSPDecoder
from .head import UnetHead
from .core import _UnetSpec, _ResnetUnetSpec


class PSPnet(_UnetSpec):

    def _get_components(
        self,
        encoder: UnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = None
        decoder = PSPDecoder(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            **decoder_params
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            upsample_scale=decoder.downsample_factor,
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head


class ResnetPSPnet(_ResnetUnetSpec):

    def _get_components(
        self,
        encoder: ResnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        bridge = None
        decoder = PSPDecoder(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            upsample_scale=decoder.downsample_factor,
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head
