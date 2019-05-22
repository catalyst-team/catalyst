from typing import List
import torch
import torch.nn as nn

from .encoder import EncoderSpec, UnetEncoder
from .bridge import BridgeSpec, UnetBridge
from .decoder import DecoderSpec, UNetDecoder
from .head import HeadSpec, UnetHead


class MetaUnet(nn.Module):
    def __init__(
        self,
        encoder: EncoderSpec,
        bridge: BridgeSpec,
        decoder: DecoderSpec,
        head: HeadSpec
    ):
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features: List[torch.Tensor] = self.encoder(x)
        bridge_features: List[torch.Tensor] = self.bridge(encoder_features)
        decoder_features: List[torch.Tensor] = self.decoder(bridge_features)
        output: torch.Tensor = self.head(decoder_features)
        return output


class UNet(MetaUnet):
    def __init__(
        self, num_classes=1, in_channels=3, num_channels=64, num_blocks=4
    ):
        encoder = UnetEncoder(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks
        )
        bridge = UnetBridge(
            in_channels=encoder.out_channels,
            out_channels=encoder.out_channels[-1] * 2
        )
        decoder = UNetDecoder(
            in_channels=bridge.out_channels,
            dilation_factors=encoder.out_strides
        )
        head = UnetHead(num_channels, num_classes)
        super().__init__(
            encoder=encoder,
            bridge=bridge,
            decoder=decoder,
            head=head
        )
