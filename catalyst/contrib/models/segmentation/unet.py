from typing import List
from functools import partial

from .blocks import UnetDownsampleBlock, UnetUpsampleBlock, UnetDecoderBlock

from .encoder import UnetEncoder, ResnetEncoder
from .bridge import BaseUnetBridge
from .decoder import UNetDecoder
from .head import BaseUnetHead
from .core import UnetSpec


class UNet(UnetSpec):
    def __init__(
        self, num_classes=1, in_channels=3, num_channels=32, num_blocks=4
    ):
        encoder = UnetEncoder(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks
        )
        bridge = BaseUnetBridge(
            in_channels=encoder.out_channels,
            out_channels=encoder.out_channels[-1] * 2,
            block_fn=UnetDownsampleBlock
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            dilation_factors=encoder.out_strides,
            block_fn=UnetDecoderBlock
        )
        head = BaseUnetHead(decoder.out_channels[-1], num_classes)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )


class ResnetUnet(UnetSpec):

    def __init__(
        self,
        num_classes=1,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = False,
        layers: List[int] = None
    ):
        encoder = ResnetEncoder(
            arch=arch,
            pretrained=pretrained,
            requires_grad=requires_grad,
            layers=layers
        )
        bridge = BaseUnetBridge(
            in_channels=encoder.out_channels,
            out_channels=encoder.out_channels[-1],
            block_fn=partial(UnetUpsampleBlock, pool_first=True)
        )
        decoder_in_channels = encoder.out_channels \
            if bridge is None \
            else bridge.out_channels
        decoder = UNetDecoder(
            in_channels=decoder_in_channels,
            block_fn=partial(
                UnetDecoderBlock,
                cat_first=True,
                upsample_scale=2)
            # dilation_factors=encoder.out_strides
        )
        head = BaseUnetHead(
            decoder.out_channels[-1],
            num_classes,
            upsample=1)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )
