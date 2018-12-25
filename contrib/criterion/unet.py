import torch
from torch import nn
from torch.nn import functional as F


class LossBinary(nn.Module):
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection
    using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        super().__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log(
                (intersection + eps) / (union - intersection + eps)
            )
        return loss


class LossMulti(nn.Module):
    def __init__(self, jaccard_weight=0, num_classes=1, class_weights=None):
        super().__init__()
        self.nll_loss = nn.NLLLoss(weight=class_weights)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        nll_loss = self.nll_loss(outputs, targets)
        jaccard_loss = []
        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()

                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum() + eps
                jaccard_loss.append(1 - intersection / (union - intersection))

        return nll_loss, jaccard_loss
