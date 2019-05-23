from typing import List
from functools import partial

from .blocks import UnetUpsampleBlock, LinknetDecoderBlock

from .encoder import UnetEncoder, ResnetEncoder
from .bridge import BaseUnetBridge
from .decoder import UNetDecoder
from .head import BaseUnetHead
from .core import UnetSpec


class Linknet(UnetSpec):
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
            out_channels=encoder.out_channels[-1] * 2
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            dilation_factors=encoder.out_strides,
            block_fn=LinknetDecoderBlock
        )
        head = BaseUnetHead(num_channels, num_classes)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )


class ResnetLinknet(UnetSpec):

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
        bridge = None
        # bridge = BaseUnetBridge(
        #     in_channels=encoder.out_channels,
        #     out_channels=encoder.out_channels[-1],
        #     block_fn=partial(UnetUpsampleBlock, pool_first=False)
        # )
        decoder_in_channels = encoder.out_channels \
            if bridge is None \
            else bridge.out_channels
        decoder = UNetDecoder(
            in_channels=decoder_in_channels,
            block_fn=partial(
                LinknetDecoderBlock,
                sum_first=False,
                upsample_scale=None)
            # dilation_factors=encoder.out_strides
        )
        head = BaseUnetHead(
            decoder.out_channels[-1],
            num_classes,
            upsample=2)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )
