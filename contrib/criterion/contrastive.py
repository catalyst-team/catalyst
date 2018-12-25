import torch
import torch.nn as nn


class ContrastiveEmbeddingLoss(nn.Module):
    """
    Contrastive embedding loss

    paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0, reduction="elementwise_mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, x0, x1, y):
        """
        :param x0:
        :param x1:
        :param y: 0 if same class else 1
        :return:
        """
        # euclidian distance
        diff = x0 - x1
        dist = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))

        bs = len(y)
        mdist = self.margin - dist
        mdist_ = torch.clamp(mdist, min=0.0)
        loss = (1 - y) * torch.pow(dist, 2) + y * torch.pow(mdist_, 2)

        if self.reduction == "elementwise_mean":
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class ContrastiveDistanceLoss(nn.Module):
    """
    Contrastive distance loss
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, y):
        bs = len(y)
        mdist = self.margin - dist
        mdist_ = torch.clamp(mdist, min=0.0)
        loss = (1 - y) * torch.pow(dist, 2) + y * torch.pow(mdist_, 2)
        loss = torch.sum(loss) / 2.0 / bs
        return loss
