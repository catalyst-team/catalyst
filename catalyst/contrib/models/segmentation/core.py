from typing import List
import torch
import torch.nn as nn

from .encoder import EncoderSpec
from .bridge import BridgeSpec
from .decoder import DecoderSpec
from .head import HeadSpec


class UnetSpec(nn.Module):
    def __init__(
        self,
        encoder: EncoderSpec,
        decoder: DecoderSpec,
        bridge: BridgeSpec = None,
        head: HeadSpec = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge or (lambda x: x)
        self.decoder = decoder
        self.head = head or (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"x: {[z.shape for z in [x]]}")
        encoder_features: List[torch.Tensor] = self.encoder(x)
        print(f"encoder_features: {[z.shape for z in encoder_features]}")
        bridge_features: List[torch.Tensor] = self.bridge(encoder_features)
        print(f"bridge_features: {[z.shape for z in bridge_features]}")
        decoder_features: List[torch.Tensor] = self.decoder(bridge_features)
        print(f"decoder_features: {[z.shape for z in decoder_features]}")
        output: torch.Tensor = self.head(decoder_features)
        print(f"output: {[z.shape for z in [output]]}")
        return output
