# flake8: noqa

from torch.nn.modules.loss import *
from .ce import NaiveCrossEntropyLoss
from .center import CenterLoss
from .contrastive import ContrastiveDistanceLoss, ContrastiveEmbeddingLoss
from .dice import DiceLoss, BCEDiceLoss
from .focal import FocalLossBinary, FocalLossMultiClass
from .huber import HuberLoss
from .iou import IoULoss, BCEIoULoss
from .lovasz import LovaszLossBinary, LovaszLossMultiClass, \
    LovaszLossMultiLabel
from .wing import WingLoss
