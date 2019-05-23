import torch
import torch.nn as nn
import torch.nn.functional as F

from ..abn import ABN, ACT_RELU
from .core import EncoderBlock, CentralBlock, DecoderBlock, \
    _get_block, _upsample


class UnetEncoderBlock(EncoderBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        stride: int = 1,
        **kwargs
    ):
        super().__init__()
        self._block = _get_block(
            in_channels=in_channels,
            out_channels=out_channels,
            abn_block=abn_block,
            activation=activation,
            first_stride=1,
            second_stride=stride,
            **kwargs
        )

    @property
    def block(self):
        return self._block


class UnetDownsampleBlock(CentralBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        **kwargs
    ):
        super().__init__()
        self._block = _get_block(
            in_channels=in_channels,
            out_channels=out_channels,
            abn_block=abn_block,
            activation=activation,
            first_stride=2,
            second_stride=1,
            **kwargs
        )

    @property
    def block(self):
        return self._block


class UnetUpsampleBlock(CentralBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        pool_first: bool = False,
        **kwargs
    ):
        super().__init__()
        self.pool_first = pool_first
        self._block = _get_block(
            in_channels=in_channels,
            out_channels=out_channels,
            abn_block=abn_block,
            activation=activation,
            first_stride=1,
            second_stride=1,
            **kwargs
        )

    @property
    def block(self):
        return self._block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_first:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(x)


class UnetDecoderBlock(DecoderBlock):
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
        cat_first: bool = False,
        **kwargs
    ):
        super().__init__(in_channels, enc_channels, out_channels)
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.cat_first = cat_first

        self._block = nn.Sequential(
            nn.Dropout(pre_dropout_rate, inplace=True),
            _get_block(
                in_channels=in_channels + enc_channels,
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
        down: torch.Tensor,
        left: torch.Tensor
    ) -> torch.Tensor:

        if self.cat_first:
            x = torch.cat([down, left], 1)
            x = _upsample(
                x,
                scale=self.upsample_scale,
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners
            )
        else:
            x = _upsample(
                down,
                scale=self.upsample_scale,
                size=left.shape[2:],
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners
            )
            x = torch.cat([x, left], 1)

        return self.block(x)
