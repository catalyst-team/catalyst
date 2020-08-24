# flake8: noqa
from catalyst.contrib.models.cv.segmentation.blocks.core import (
    DecoderBlock,
    EncoderBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.fpn import (
    Conv3x3GNReLU,
    DecoderFPNBlock,
    SegmentationBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.psp import (
    PSPBlock,
    PyramidBlock,
)
from catalyst.contrib.models.cv.segmentation.blocks.unet import (
    DecoderConcatBlock,
    DecoderSumBlock,
    EncoderDownsampleBlock,
    EncoderUpsampleBlock,
)
