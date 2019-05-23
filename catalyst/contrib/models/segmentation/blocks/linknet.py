import torch
import torch.nn as nn

from .core import DecoderBlock, _get_block, _upsample
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
            _get_block(
                in_channels=in_channels,
                out_channels=out_channels,
                abn_block=abn_block,
                activation=activation,
                first_stride=1,
                second_stride=1,
                **kwargs
            ),
            nn.Dropout(post_dropout_rate, inplace=True)
        )

    @property
    def block(self):
        return self._block

    def forward(
        self,
        bottom: torch.Tensor,
        left: torch.Tensor
    ) -> torch.Tensor:

        if self.sum_first:
            x = bottom + left
            x = _upsample(
                x,
                scale=self.upsample_scale,
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners
            )
            x = self.block(x)
        else:
            x = _upsample(
                bottom,
                scale=self.upsample_scale,
                size=left.shape[2:],
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners
            )
            x = self.block(x)
            x = x + left

        return x
