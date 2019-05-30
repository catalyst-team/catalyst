from typing import List
from functools import partial

import torch

from .core import DecoderSpec
from ..blocks.core import _get_block
from ..blocks.psp import PSPBlock
from ..abn import ABN


class PSPDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        downsample_factor: int = 8,
        use_batchnorm: bool = True,
        out_channels: int = 512
    ):
        super().__init__(in_channels, in_strides)
        self.block_offset = self._get_block_offset(downsample_factor)
        psp_out_channels: int = self._get(in_channels)

        self.psp = PSPBlock(
            psp_out_channels,
            pool_sizes=(1, 2, 3, 6),
            use_batchnorm=use_batchnorm,
        )

        self.conv = _get_block(
            psp_out_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            abn_block=partial(ABN, use_batchnorm=use_batchnorm),
            complexity=0,
        )
        self._out_channels = out_channels
        self.downsample_factor = downsample_factor

    @property
    def out_channels(self) -> List[int]:
        return [self._out_channels]

    @property
    def out_strides(self) -> List[int]:
        return [self.downsample_factor]

    def _get_block_offset(self, downsample_factor: int):
        offset = self.in_strides.index(downsample_factor)
        return offset

    def _get(self, xs: List):
        return xs[self.block_offset]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        features = self._get(x)
        x = self.psp(features)
        x = self.conv(x)
        return [x]
