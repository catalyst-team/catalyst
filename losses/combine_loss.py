import torch
import torch.nn as nn
import torch.nn.functional as F
from .f1 import F1Loss
from .focal_loss import FocalLoss


class BCEAndF1(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEAndF1, self).__init__()
        self.bce_weight = bce_weight
        self.f1_loss = F1Loss()
        self.bce_loss = F.binary_cross_entropy_with_logits

    def forward(self, logits, labels):
        f1 = self.f1_loss(logits, labels)
        bce = self.bce_loss(logits, labels)
        return self.bce_weight * bce + (1 - self.bce_weight) * f1


class BCEF1FocalLoss(nn.Module):
    def __init__(self):
        super(BCEF1FocalLoss, self).__init__()
        self.f1_loss = F1Loss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.focal_loss = FocalLoss()

    def forward(self, logits, labels):
        f1 = self.f1_loss(logits, labels)
        bce = self.bce_loss(logits, labels)
        focal = self.focal_loss(logits, labels)
        return 0.45 * bce + 0.45 * f1 + 0.1 * focal
