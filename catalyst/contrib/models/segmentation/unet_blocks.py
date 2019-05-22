import torch
import torch.nn as nn
import torch.nn.functional as F

from .abn import ABN, ACT_RELU


class UnetEncoderBlock(nn.Module):
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
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, stride=1, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, stride=stride, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation)
        )

    def forward(self, x):
        return self.block(x)


class UnetCentralBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, stride=2, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
                **kwargs),
            abn_block(out_channels, activation=activation)
        )

    def forward(self, x):
        return self.block(x)


class UnetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        abn_block: nn.Module = ABN,
        activation: str = ACT_RELU,
        pre_dropout_rate: float = 0.,
        post_dropout_rate: float = 0.,
        **kwargs
    ):
        super().__init__()

        self.block = nn.Sequential(
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

    def forward(self, down, left):
        encoder_hw = left.size()[2:]
        x = F.interpolate(
            down, size=encoder_hw, mode="bilinear", align_corners=True)
        x = torch.cat([x, left], 1)
        return self.block(x)
