from typing import List

import torch

from .core import DecoderSpec
from ..blocks.core import _get_block
from ..blocks.psp import PSPBlock
from ..abn import ABN_fake, ABN


class PSPDecoder(DecoderSpec):
    def __init__(
        self,
        in_channels: List[int],
        downsample_factor: int = 8,
        use_batchnorm: bool = True,
        out_channels: int = 512,
        block_offset: int = 0,
    ):
        super().__init__()
        self.block_offset = self._get_block_offset(
            downsample_factor,
            block_offset)
        psp_out_channels: int = self._get(in_channels)

        self.psp = PSPBlock(
            psp_out_channels,
            pool_sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = _get_block(
            psp_out_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            abn_block=ABN if use_batchnorm else ABN_fake,
            complexity=0,
        )
        self._out_channels = out_channels
        self.downsample_factor = downsample_factor

    @property
    def out_channels(self) -> List[int]:
        return [self._out_channels]

    def _get_block_offset(self, downsample_factor: int, block_offset: int = 0):
        if downsample_factor == 4:
            offset = 1
        elif downsample_factor == 8:
            offset = 2
        elif downsample_factor == 16:
            offset = 3
        else:
            raise ValueError(
                f"Downsample factor should bi in [4, 8, 16],"
                f" got {downsample_factor}")
        return block_offset + offset

    def _get(self, xs: List):
        return xs[self.block_offset]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        features = self._get(x)
        x = self.psp(features)
        x = self.conv(x)
        return [x]
