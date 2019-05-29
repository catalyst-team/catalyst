import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import DecoderBlock


class DecoderFPNBlock(DecoderBlock):
    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        out_channels: int,
        in_strides: int = None,
        upsample_scale: int = 2,
        interpolation_mode: str = "nearest",
        align_corners: bool = None,
        aggregate_first: bool = False,
        **kwargs
    ):
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        super().__init__(
            in_channels, enc_channels, out_channels, in_strides,
            **kwargs
        )

    def _get_block(self):
        block = nn.Conv2d(
            self.enc_channels,
            self.out_channels,
            kernel_size=1)
        return block

    def forward(
        self,
        bottom: torch.Tensor,
        left: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(
            bottom,
            scale_factor=self.upsample_scale,
            mode=self.interpolation_mode,
            align_corners=self.align_corners)
        left = self.block(left)
        x = x + left
        return x


class Conv3x3GNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample=False,
        upsample_scale: int = 2,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        super().__init__()
        self.upsample = upsample
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners)
        return x


class SegmentationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_upsamples: int = 0
    ):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(
                in_channels,
                out_channels,
                upsample=bool(num_upsamples))
        ]

        if num_upsamples > 1:
            for _ in range(1, num_upsamples):
                blocks.append(
                    Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
