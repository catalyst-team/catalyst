# flake8: noqa
from torch.nn.modules import *

from catalyst.contrib.layers.amsoftmax import AMSoftmax
from catalyst.contrib.layers.arcface import ArcFace, SubCenterArcFace
from catalyst.contrib.layers.arcmargin import ArcMarginProduct
from catalyst.contrib.layers.common import (
    Flatten,
    GaussianNoise,
    Lambda,
    Normalize,
    ResidualBlock,
)
from catalyst.contrib.layers.cosface import CosFace, AdaCos
from catalyst.contrib.layers.curricularface import CurricularFace
from catalyst.contrib.layers.factorized import FactorizedLinear
from catalyst.contrib.layers.lama import (
    LamaPooling,
    TemporalLastPooling,
    TemporalAvgPooling,
    TemporalMaxPooling,
    TemporalDropLastWrapper,
    TemporalAttentionPooling,
    TemporalConcatPooling,
)
from catalyst.contrib.layers.pooling import (
    GlobalAttnPool2d,
    GlobalAvgAttnPool2d,
    GlobalAvgPool2d,
    GlobalConcatAttnPool2d,
    GlobalConcatPool2d,
    GlobalMaxAttnPool2d,
    GlobalMaxPool2d,
    GeM2d,
)
from catalyst.contrib.layers.rms_norm import RMSNorm
from catalyst.contrib.layers.se import (
    sSE,
    scSE,
    cSE,
)
from catalyst.contrib.layers.softmax import SoftMax
