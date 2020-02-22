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
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Symmetric Cross Entropy
        paper : https://arxiv.org/abs/1908.06112

        Args:
            alpha(float):
                corresponds to overfitting issue of CE
            beta(float):
                corresponds to flexible exploration on the robustness of RCE
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        """
        Args:
            input: shape = [batch_size; num_classes]
            target: shape = [batch_size]
            values of a vector correspond to class index
        """
        num_classes = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        assert target_one_hot.shape == input.shape

        input = torch.clamp(input, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        cross_entropy = (-torch.sum(target_one_hot * torch.log(input),
                                    dim=1)).mean()
        reverse_cross_entropy = (
            -torch.sum(input * torch.log(target_one_hot), dim=1)
        ).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class MaskCrossEntropyLoss(torch.nn.CrossEntropyLoss):
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


__all__ = [
    "MaskCrossEntropyLoss",
    "SymmetricCrossEntropyLoss",
    "NaiveCrossEntropyLoss",
]
