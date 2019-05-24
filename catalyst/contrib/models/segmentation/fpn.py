from typing import List

from .encoder import UnetEncoder, ResnetEncoder
from .bridge import BaseUnetBridge
from .decoder import FPNDecoder
from .head import BaseUnetHead
from .core import UnetSpec


class FPNUnet(UnetSpec):

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ):
        encoder = UnetEncoder(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks
        )
        bridge = BaseUnetBridge(
            in_channels=encoder.out_channels,
            out_channels=encoder.out_channels[-1] * 2
        )
        decoder_in_channels = encoder.out_channels \
            if bridge is None \
            else bridge.out_channels
        decoder = FPNDecoder(
            in_channels=decoder_in_channels,
            # dilation_factors=encoder.out_strides
        )
        head = BaseUnetHead(
            decoder.out_channels[-1],
            num_classes,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )


class ResnetFPNUnet(UnetSpec):

    def __init__(
        self,
        num_classes=1,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = False,
        layers: List[int] = None,
        dropout: float = 0.0,
    ):
        encoder = ResnetEncoder(
            arch=arch,
            pretrained=pretrained,
            requires_grad=requires_grad,
            layers=layers
        )
        bridge = None
        decoder_in_channels = encoder.out_channels \
            if bridge is None \
            else bridge.out_channels
        decoder = FPNDecoder(
            in_channels=decoder_in_channels,
            # dilation_factors=encoder.out_strides
        )
        head = BaseUnetHead(
            decoder.out_channels[-1],
            num_classes,
            dropout=dropout,
            upsample_scale=4,
            interpolation_mode="bilinear",
            align_corners=True
        )
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )
