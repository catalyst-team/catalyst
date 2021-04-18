from typing import Dict

from catalyst.contrib.models.cv.segmentation.blocks import EncoderDownsampleBlock
from catalyst.contrib.models.cv.segmentation.bridge import UnetBridge
from catalyst.contrib.models.cv.segmentation.core import ResnetUnetSpec, UnetSpec
from catalyst.contrib.models.cv.segmentation.decoder import FPNDecoder
from catalyst.contrib.models.cv.segmentation.encoder import ResnetEncoder, UnetEncoder
from catalyst.contrib.models.cv.segmentation.head import FPNHead


class FPNUnet(UnetSpec):
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
        decoder = FPNDecoder(
            in_channels=bridge.out_channels, in_strides=bridge.out_strides, **decoder_params
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
        decoder = FPNDecoder(
            in_channels=encoder.out_channels, in_strides=encoder.out_strides, **decoder_params
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


__all__ = ["FPNUnet", "ResnetFPNUnet"]
