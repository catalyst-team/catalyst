from .encoder import UnetEncoder
from .bridge import BaseUnetBridge
from .decoder import UNetDecoder
from .head import BaseUnetHead
from .core import UnetSpec
from .blocks import LinknetDecoderBlock


class Linknet(UnetSpec):
    def __init__(
        self, num_classes=1, in_channels=3, num_channels=64, num_blocks=4
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
            decoder_block=LinknetDecoderBlock
        )
        head = BaseUnetHead(num_channels, num_classes)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )
