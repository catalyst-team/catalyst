import torch.nn as nn
import torch.nn.functional as F


class FPNBlock(nn.Module):
    def __init__(
        self,
        pyramid_channels: int,
        encoder_channels: int,
        upsample_scale: int = 2,
        interpolation_mode: str = "nearest",
        align_corners: bool = None,
    ):
        super().__init__()
        self.upsample_scale = upsample_scale
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self._block = nn.Conv2d(
            encoder_channels,
            pyramid_channels,
            kernel_size=1)

    def forward(self, bottom, left):
        x = F.interpolate(
            bottom,
            scale_factor=self.upsample_scale,
            mode=self.interpolation_mode,
            align_corners=self.align_corners)
        left = self._block(left)
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

        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self._block(x)
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

        self._block = nn.Sequential(*blocks)

    def forward(self, x):
        return self._block(x)
