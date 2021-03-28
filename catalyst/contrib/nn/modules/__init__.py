# flake8: noqa
from torch.nn.modules import *

from catalyst.contrib.nn.modules.amsoftmax import AMSoftmax
from catalyst.contrib.nn.modules.arcface import ArcFace, SubCenterArcFace
from catalyst.contrib.nn.modules.arcmargin import ArcMarginProduct
from catalyst.contrib.nn.modules.common import (
    Flatten,
    GaussianNoise,
    Lambda,
    Normalize,
)
from catalyst.contrib.nn.modules.cosface import CosFace, AdaCos
from catalyst.contrib.nn.modules.curricularface import CurricularFace
from catalyst.contrib.nn.modules.factorized import FactorizedLinear
from catalyst.contrib.nn.modules.lama import (
    LamaPooling,
    TemporalLastPooling,
    TemporalAvgPooling,
    TemporalMaxPooling,
    TemporalDropLastWrapper,
    TemporalAttentionPooling,
    TemporalConcatPooling,
)
from catalyst.contrib.nn.modules.pooling import (
    GlobalAttnPool2d,
    GlobalAvgAttnPool2d,
    GlobalAvgPool2d,
    GlobalConcatAttnPool2d,
    GlobalConcatPool2d,
    GlobalMaxAttnPool2d,
    GlobalMaxPool2d,
    GeM2d,
)
from catalyst.contrib.nn.modules.rms_norm import RMSNorm
from catalyst.contrib.nn.modules.se import (
    sSE,
    scSE,
    cSE,
)
from catalyst.contrib.nn.modules.softmax import SoftMax
