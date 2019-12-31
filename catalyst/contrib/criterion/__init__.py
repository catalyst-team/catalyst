# flake8: noqa
from torch.nn.modules.loss import *

from .ce import (
    MaskCrossEntropyLoss, NaiveCrossEntropyLoss, SymmetricCrossEntropyLoss
)
from .center import CenterLoss
from .contrastive import (
    ContrastiveDistanceLoss, ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss
)
from .dice import BCEDiceLoss, DiceLoss
from .focal import FocalLossBinary, FocalLossMultiClass
from .huber import HuberLoss
from .iou import BCEIoULoss, IoULoss
from .lovasz import (
    LovaszLossBinary, LovaszLossMultiClass, LovaszLossMultiLabel
)
from .triplet import TripletLoss, TripletPairwiseEmbeddingLoss
from .wing import WingLoss
