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
        upsample_scale: int = None,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
        sum_first: bool = False,
        **kwargs
    ):
        super().__init__(in_channels, enc_channels, out_channels)
        assert enc_channels == out_channels
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.sum_first = sum_first

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

    def upsample(
        self,
        x: torch.Tensor,
        scale: int = None,
        size: int = None
    ) -> torch.Tensor:
        if scale is None:
            x = F.interpolate(
                x,
                size=size,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        else:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        return x

    def forward(self, down: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        encoder_hw = left.shape[2:]
        x = F.interpolate(
            down, size=encoder_hw, mode="bilinear", align_corners=True)
        x = self.block(x)
        return x + left

    def forward(
        self,
        down: torch.Tensor,
        left: torch.Tensor
    ) -> torch.Tensor:

        if self.sum_first:
            x = down + left
            x = self.upsample(x, scale=self.upsample_scale)
            x = self.block(x)
        else:
            x = self.upsample(
                down,
                scale=self.upsample_scale,
                size=left.shape[2:])
            x = self.block(x)
            x = x + left

        return x
