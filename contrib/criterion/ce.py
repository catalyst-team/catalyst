import torch
import torch.nn as nn
import torch.nn.functional as F


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
