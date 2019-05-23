import torch
import torch.nn.functional as F
import torch.nn as nn

from .core import DecoderBlock
from ..abn import ABN, ACT_RELU


class LinknetDecoderBlock(DecoderBlock):
    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        pre_dropout_rate: float = 0.,
        post_dropout_rate: float = 0.,
        **kwargs
    ):
        super().__init__(in_channels, enc_channels, out_channels)
        assert enc_channels == out_channels

        self._block = nn.Sequential(
            nn.Dropout(pre_dropout_rate, inplace=True),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, stride=1, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, stride=1, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation),
            nn.Dropout(post_dropout_rate, inplace=True)
        )

    @property
    def block(self):
        return self._block

    def forward(self, down: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        encoder_hw = left.shape[2:]
        x = F.interpolate(
            down, size=encoder_hw, mode="bilinear", align_corners=True)
        x = self.block(x)
        return x + left
