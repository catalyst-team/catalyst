import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEmbeddingLoss(nn.Module):
    """
    Contrastive embedding loss

    paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))

        bs = len(y)
        mdist = self.margin - dist
        mdist_ = torch.clamp(mdist, min=0.0)
        loss = (1 - y) * torch.pow(dist, 2) + y * torch.pow(mdist_, 2)

        if self.reduction == "mean":
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


class ContrastivePairwiseEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, embeddings_pred, embeddings_true):
        device = embeddings_pred.device
        # s - state space
        # d - embeddings space
        # a - action space
        pairwise_similarity = torch.einsum(
            "se,ae->sa",
            embeddings_pred,
            embeddings_true
        )
        bs = embeddings_pred.shape[0]
        batch_idx = torch.arange(bs, device=device)
        loss = F.cross_entropy(
            pairwise_similarity,
            batch_idx,
            reduction=self.reduction
        )
        return loss


__all__ = [
    "ContrastiveEmbeddingLoss",
    "ContrastiveDistanceLoss",
    "ContrastivePairwiseEmbeddingLoss"
]
