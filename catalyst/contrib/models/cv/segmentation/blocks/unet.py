# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import torch
from torch import nn
from torch.nn import functional as F

from catalyst.contrib.models.cv.segmentation.abn import ABN
from catalyst.contrib.models.cv.segmentation.blocks.core import (  # noqa: WPS450, E501
    _get_block,
    _upsample,
    DecoderBlock,
    EncoderBlock,
)


class EncoderDownsampleBlock(EncoderBlock):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_strides: int = None,
        abn_block: nn.Module = ABN,
        activation: str = "ReLU",
        first_stride: int = 2,
        second_stride: int = 1,
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(in_channels, out_channels, in_strides)
        self._out_strides = (
            in_strides * first_stride * second_stride if in_strides is not None else None
        )
        self._block = _get_block(
            in_channels=in_channels,
            out_channels=out_channels,
            abn_block=abn_block,
            activation=activation,
            first_stride=first_stride,
            second_stride=second_stride,
            **kwargs
        )

    @property
    def out_strides(self) -> int:
        """@TODO: Docs. Contribution is welcome."""
        return self._out_strides

    @property
    def block(self):
        """@TODO: Docs. Contribution is welcome."""
        return self._block


class EncoderUpsampleBlock(EncoderBlock):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_strides: int = None,
        abn_block: nn.Module = ABN,
        activation: str = "ReLU",
        first_stride: int = 1,
        second_stride: int = 1,
        pool_first: bool = False,
        upsample_scale: int = 2,
        interpolation_mode: str = "nearest",
        align_corners: bool = None,
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(in_channels, out_channels, in_strides)
        if in_strides is None:
            self._out_strides = None
        elif pool_first:
            self._out_strides = in_strides * first_stride * second_stride * 2 // upsample_scale
        else:
            self._out_strides = in_strides * first_stride * second_stride // upsample_scale
        self.pool_first = pool_first
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self._block = _get_block(
            in_channels=in_channels,
            out_channels=out_channels,
            abn_block=abn_block,
            activation=activation,
            first_stride=first_stride,
            second_stride=second_stride,
            **kwargs
        )

    @property
    def out_strides(self) -> int:
        """@TODO: Docs. Contribution is welcome."""
        return self._out_strides

    @property
    def block(self):
        """@TODO: Docs. Contribution is welcome."""
        return self._block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        if self.pool_first:
            x = F.max_pool2d(x, kernel_size=self.upsample_scale, stride=self.upsample_scale)
        x = F.interpolate(
            x,
            scale_factor=self.upsample_scale,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
        )
        return self.block(x)


class DecoderConcatBlock(DecoderBlock):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        out_channels: int,
        in_strides: int = None,
        abn_block: nn.Module = ABN,
        activation: str = "ReLU",
        pre_dropout_rate: float = 0.0,
        post_dropout_rate: float = 0.0,
        upsample_scale: int = None,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
        aggregate_first: bool = False,
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.aggregate_first = aggregate_first

        super().__init__(
            in_channels,
            enc_channels,
            out_channels,
            in_strides,
            abn_block=abn_block,
            activation=activation,
            pre_dropout_rate=pre_dropout_rate,
            post_dropout_rate=post_dropout_rate,
            **kwargs
        )

    def _get_block(
        self,
        abn_block: nn.Module = ABN,
        activation: str = "ReLU",
        pre_dropout_rate: float = 0.0,
        post_dropout_rate: float = 0.0,
        **kwargs
    ):
        layers = []
        if pre_dropout_rate > 0:
            layers.append(nn.Dropout2d(pre_dropout_rate, inplace=True))
        layers.append(
            _get_block(
                in_channels=self.in_channels + self.enc_channels,
                out_channels=self.out_channels,
                abn_block=abn_block,
                activation=activation,
                first_stride=1,
                second_stride=1,
                **kwargs
            )
        )
        if post_dropout_rate > 0:
            layers.append(nn.Dropout2d(pre_dropout_rate, inplace=True))

        block = nn.Sequential(*layers)
        return block

    def forward(self, bottom: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        if self.aggregate_first:
            x = torch.cat([bottom, left], 1)
            x = _upsample(
                x,
                scale=self.upsample_scale,
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
        else:
            x = _upsample(
                bottom,
                scale=self.upsample_scale,
                size=left.shape[2:],
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
            x = torch.cat([x, left], 1)

        return self.block(x)


class DecoderSumBlock(DecoderConcatBlock):
    """@TODO: Docs (add description, `Example`). Contribution is welcome"""

    def __init__(self, enc_channels: int, **kwargs):
        """
        Args:
            @TODO: Docs. Contribution is welcome.
        """
        super().__init__(enc_channels=0, **kwargs)

    def forward(self, bottom: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        if self.aggregate_first:
            x = bottom + left
            x = _upsample(
                x,
                scale=self.upsample_scale,
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
            x = self.block(x)
        else:
            x = _upsample(
                bottom,
                scale=self.upsample_scale,
                size=left.shape[2:],
                interpolation_mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
            x = self.block(x)
            x = x + left

        return x


__all__ = [
    "EncoderDownsampleBlock",
    "EncoderUpsampleBlock",
    "DecoderConcatBlock",
    "DecoderSumBlock",
]
