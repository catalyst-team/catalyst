# flake8: noqa

from torch.nn.modules.loss import *

from catalyst.contrib.nn.criterion.ce import (
    MaskCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
    NaiveCrossEntropyLoss,
)
from catalyst.contrib.nn.criterion.circle import CircleLoss
from catalyst.contrib.nn.criterion.contrastive import (
    ContrastiveDistanceLoss,
    ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss,
)
from catalyst.contrib.nn.criterion.dice import DiceLoss
from catalyst.contrib.nn.criterion.focal import (
    FocalLossBinary,
    FocalLossMultiClass,
)
from catalyst.contrib.nn.criterion.gan import (
    GradientPenaltyLoss,
    MeanOutputLoss,
)
from catalyst.contrib.nn.criterion.huber import HuberLoss
from catalyst.contrib.nn.criterion.iou import IoULoss
from catalyst.contrib.nn.criterion.trevsky import TrevskyLoss, FocalTrevskyLoss
from catalyst.contrib.nn.criterion.lovasz import (
    LovaszLossBinary,
    LovaszLossMultiClass,
    LovaszLossMultiLabel,
)
from catalyst.contrib.nn.criterion.margin import MarginLoss
from catalyst.contrib.nn.criterion.triplet import (
    TripletLoss,
    TripletLossV2,
    TripletPairwiseEmbeddingLoss,
    TripletMarginLossWithSampler,
)
from catalyst.contrib.nn.criterion.wing import WingLoss
