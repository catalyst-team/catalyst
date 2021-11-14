# flake8: noqa

import torch
from torch.nn.modules.loss import *

from catalyst.contrib.losses.ce import (
    MaskCrossEntropyLoss,
    NaiveCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
)
from catalyst.contrib.losses.circle import CircleLoss
from catalyst.contrib.losses.contrastive import (
    BarlowTwinsLoss,
    ContrastiveDistanceLoss,
    ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss,
)
from catalyst.contrib.losses.dice import DiceLoss
from catalyst.contrib.losses.focal import FocalLossBinary, FocalLossMultiClass
from catalyst.contrib.losses.gan import GradientPenaltyLoss, MeanOutputLoss

from catalyst.contrib.losses.iou import IoULoss
from catalyst.contrib.losses.lovasz import (
    LovaszLossBinary,
    LovaszLossMultiClass,
    LovaszLossMultiLabel,
)
from catalyst.contrib.losses.margin import MarginLoss
from catalyst.contrib.losses.ntxent import NTXentLoss
from catalyst.contrib.losses.recsys import (
    AdaptiveHingeLoss,
    BPRLoss,
    HingeLoss,
    LogisticLoss,
    RocStarLoss,
    WARPLoss,
)
from catalyst.contrib.losses.regression import (
    HuberLossV0,
    CategoricalRegressionLoss,
    QuantileRegressionLoss,
    RSquareLoss,
)
from catalyst.contrib.losses.supervised_contrastive import SupervisedContrastiveLoss
from catalyst.contrib.losses.smoothing_dice import SmoothingDiceLoss
from catalyst.contrib.losses.trevsky import FocalTrevskyLoss, TrevskyLoss
from catalyst.contrib.losses.triplet import (
    TripletLoss,
    TripletLossV2,
    TripletMarginLossWithSampler,
    TripletPairwiseEmbeddingLoss,
)
from catalyst.contrib.losses.wing import WingLoss
