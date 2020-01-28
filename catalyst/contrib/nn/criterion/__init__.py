# flake8: noqa
from torch.nn.modules.loss import *

from .ce import (
    MaskCrossEntropyLoss, NaiveCrossEntropyLoss, SymmetricCrossEntropyLoss
)
from .contrastive import (
    ContrastiveDistanceLoss, ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss
)
from .dice import BCEDiceLoss, DiceLoss
from .focal import FocalLossBinary, FocalLossMultiClass
from .gan import GradientPenaltyLoss, MeanOutputLoss
from .huber import HuberLoss
from .iou import BCEIoULoss, IoULoss
from .lovasz import (
    LovaszLossBinary, LovaszLossMultiClass, LovaszLossMultiLabel
)
from .margin import MarginLoss
from .triplet import TripletLoss, TripletLossV2, TripletPairwiseEmbeddingLoss
from .wing import WingLoss
