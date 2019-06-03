from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import _get_block
from ..abn import ABN


class PyramidBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int,
        use_batchnorm: bool = True,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
        complexity: int = 0
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        if pool_size == 1:
            use_batchnorm = False

        self._block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            _get_block(
                in_channels,
                out_channels,
                abn_block=partial(ABN, use_batchnorm=use_batchnorm),
                complexity=complexity)
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        x = self._block(x)
        x = F.interpolate(
            x,
            size=(h, w),
            mode=self.interpolation_mode,
            align_corners=self.align_corners)
        return x


class PSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_sizes: Tuple[int] = (1, 2, 3, 6),
        use_batchnorm: bool = True
    ):
        super().__init__()

        self.stages = nn.ModuleList([
            PyramidBlock(
                in_channels, in_channels // len(pool_sizes),
                pool_size,
                use_batchnorm=use_batchnorm)
            for pool_size in pool_sizes
        ])

    def forward(self, x):
        xs = [stage(x) for stage in self.stages] + [x]
        x = torch.cat(xs, dim=1)
        return x
