# flake8: noqa
from catalyst.contrib.models.cv.segmentation.blocks.core import (
    EncoderBlock,
    DecoderBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.fpn import (
    DecoderFPNBlock,
    Conv3x3GNReLU,
    SegmentationBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.psp import (
    PyramidBlock,
    PSPBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.unet import (
    EncoderDownsampleBlock,
    EncoderUpsampleBlock,
    DecoderConcatBlock,
    DecoderSumBlock,
)
