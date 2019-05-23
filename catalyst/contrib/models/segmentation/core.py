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
