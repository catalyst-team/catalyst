import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEmbeddingLoss(nn.Module):
    """
    Contrastive embedding loss

    paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0, reduction="mean"):
        """
        Constructor method for the ContrastiveEmbeddingLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, embeddings_left, embeddings_right, distance_true):
        """
        Forward propagation method for the contrastive loss.
        Args:
            embeddings_left: left objects embeddings
            embeddings_right: right objects embeddings
            distance_true: true distances

        Returns:
            loss
        """
        # euclidian distance
        diff = embeddings_left - embeddings_right
        distance_pred = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))

        bs = len(distance_true)
        margin_distance = self.margin - distance_pred
        margin_distance_ = torch.clamp(margin_distance, min=0.0)
        loss = (
            (1 - distance_true) * torch.pow(distance_pred, 2) +
            distance_true * torch.pow(margin_distance_, 2)
        )

        if self.reduction == "mean":
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class ContrastiveDistanceLoss(nn.Module):
    """
    Contrastive distance loss
    """
    def __init__(self, margin=1.0, reduction="mean"):
        """
        Constructor method for the ContrastiveDistanceLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, distance_pred, distance_true):
        """
        Forward propagation method for the contrastive loss.
        Args:
            distance_pred: predicted distances
            distance_true: true distances

        Returns:
            loss
        """
        bs = len(distance_true)
        margin_distance = self.margin - distance_pred
        margin_distance_ = torch.clamp(margin_distance, min=0.0)
        loss = (
            (1 - distance_true) * torch.pow(distance_pred, 2) +
            distance_true * torch.pow(margin_distance_, 2)
        )

        if self.reduction == "mean":
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class ContrastivePairwiseEmbeddingLoss(nn.Module):
    """
    ContrastivePairwiseEmbeddingLoss – proof of concept criterion.
    Still work in progress.
    """
    def __init__(self, margin=1.0, reduction="mean"):
        """
        Constructor method for the ContrastivePairwiseEmbeddingLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, embeddings_pred, embeddings_true):
        """
        Work in progress.
        Args:
            embeddings_pred: predicted embeddings
            embeddings_true: true embeddings

        Returns:
            loss
        """
        device = embeddings_pred.device
        # s - state space
        # d - embeddings space
        # a - action space
        pairwise_similarity = torch.einsum(
            "se,ae->sa", embeddings_pred, embeddings_true
        )
        bs = embeddings_pred.shape[0]
        batch_idx = torch.arange(bs, device=device)
        loss = F.cross_entropy(
            pairwise_similarity, batch_idx, reduction=self.reduction
        )
        return loss


__all__ = [
    "ContrastiveEmbeddingLoss", "ContrastiveDistanceLoss",
    "ContrastivePairwiseEmbeddingLoss"
]
