# flake8: noqa
from torch import nn

from catalyst.contrib import ResidualBlock


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def resnet9(in_size: int, in_channels: int, out_features: int = 512, size: int = 16):
    sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
    assert in_size >= 32, "The graph is not valid for images with resolution lower then 32x32."
    out_size = (((in_size // 32) * 32) ** 2 * 2) // size
    return nn.Sequential(
        conv_block(in_channels, sz),
        conv_block(sz, sz2, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
        conv_block(sz2, sz4, pool=True),
        conv_block(sz4, sz8, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
        nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(out_size, out_features)
        ),
    )