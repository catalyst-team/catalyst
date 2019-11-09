# flake8: noqa

from torch.nn.modules.loss import *

from .ce import NaiveCrossEntropyLoss
from .center import CenterLoss
from .contrastive import ContrastiveDistanceLoss, ContrastiveEmbeddingLoss
from .dice import BCEDiceLoss, DiceLoss
from .focal import FocalLossBinary, FocalLossMultiClass
from .huber import HuberLoss
from .iou import BCEIoULoss, IoULoss
from .lovasz import (
    LovaszLossBinary, LovaszLossMultiClass, LovaszLossMultiLabel
)
from .triplet import TripletLoss
from .wing import WingLoss
