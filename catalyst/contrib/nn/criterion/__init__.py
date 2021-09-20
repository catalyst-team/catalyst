# flake8: noqa

import torch
from torch.nn.modules.loss import *

from catalyst.contrib.nn.criterion.ce import (
    MaskCrossEntropyLoss,
    NaiveCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
)
from catalyst.contrib.nn.criterion.circle import CircleLoss
from catalyst.contrib.nn.criterion.contrastive import (
    BarlowTwinsLoss,
    ContrastiveDistanceLoss,
    ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss,
)
from catalyst.contrib.nn.criterion.dice import DiceLoss
from catalyst.contrib.nn.criterion.focal import FocalLossBinary, FocalLossMultiClass
from catalyst.contrib.nn.criterion.gan import GradientPenaltyLoss, MeanOutputLoss

from catalyst.contrib.nn.criterion.iou import IoULoss
from catalyst.contrib.nn.criterion.lovasz import (
    LovaszLossBinary,
    LovaszLossMultiClass,
    LovaszLossMultiLabel,
)
from catalyst.contrib.nn.criterion.margin import MarginLoss
from catalyst.contrib.nn.criterion.ntxent import NTXentLoss
from catalyst.contrib.nn.criterion.recsys import (
    AdaptiveHingeLoss,
    BPRLoss,
    HingeLoss,
    LogisticLoss,
    RocStarLoss,
    WARPLoss,
)
from catalyst.contrib.nn.criterion.regression import (
    HuberLossV0,
    CategoricalRegressionLoss,
    QuantileRegressionLoss,
)
from catalyst.contrib.nn.criterion.supervised_contrastive import SupervisedContrastiveLoss
from catalyst.contrib.nn.criterion.trevsky import FocalTrevskyLoss, TrevskyLoss
from catalyst.contrib.nn.criterion.triplet import (
    TripletLoss,
    TripletLossV2,
    TripletMarginLossWithSampler,
    TripletPairwiseEmbeddingLoss,
)
from catalyst.contrib.nn.criterion.wing import WingLoss
