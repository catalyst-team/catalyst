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


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, input, target):
        target_one_hot = F.one_hot(target, self.num_classes).float()
        assert target_one_hot.shape == input.shape

        input = torch.clamp(input, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        cross_entropy = (
            -torch.sum(target_one_hot * torch.log(input), dim=1)
        ).mean()
        reverse_cross_entropy = (
            -torch.sum(input * torch.log(target_one_hot), dim=1)
        ).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss
