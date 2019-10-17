import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class NaiveCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        assert input.size() == target.size()
        input = F.log_softmax(input)
        loss = -torch.sum(input * target)
        loss = loss / input.size()[0] if self.size_average else loss
        return loss


class MaskCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        *args,
        target_name: str = "targets",
        mask_name: str = "mask",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_name = target_name
        self.mask_name = mask_name
        self.reduction = "none"

    def forward(self, input, target_mask):
        target = target_mask[self.target_name]
        mask = target_mask[self.mask_name]

        loss = super().forward(input, target)
        loss = torch.mean(loss[mask == 1])
        return loss


__all__ = ["NaiveCrossEntropyLoss", "MaskCrossEntropyLoss"]
