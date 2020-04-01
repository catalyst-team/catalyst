from typing import Dict

from .core import ResnetUnetSpec, UnetSpec
from .decoder import PSPDecoder
from .encoder import ResnetEncoder, UnetEncoder
from .head import UnetHead


class PSPnet(UnetSpec):
    """@TODO: Docs. Contribution is welcome."""

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
            upsample_scale=decoder.out_strides[-1],
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head


class ResnetPSPnet(ResnetUnetSpec):
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
        decoder = PSPDecoder(
            in_channels=encoder.out_channels,
            in_strides=encoder.out_strides,
            **decoder_params
        )
        head = UnetHead(
            in_channels=decoder.out_channels,
            in_strides=decoder.out_strides,
            out_channels=num_classes,
            upsample_scale=decoder.out_strides[-1],
            interpolation_mode="bilinear",
            align_corners=True,
            **head_params
        )
        return encoder, bridge, decoder, head
