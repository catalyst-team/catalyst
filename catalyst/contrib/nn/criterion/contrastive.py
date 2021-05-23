import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveEmbeddingLoss(nn.Module):
    """The Contrastive embedding loss.

    It has been proposed in `Dimensionality Reduction
    by Learning an Invariant Mapping`_.

    .. _Dimensionality Reduction by Learning an Invariant Mapping:
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """
        Args:
            margin: margin parameter
            reduction: criterion reduction type
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(
        self, embeddings_left: torch.Tensor, embeddings_right: torch.Tensor, distance_true,
    ) -> torch.Tensor:
        """Forward propagation method for the contrastive loss.

        Args:
            embeddings_left: left objects embeddings
            embeddings_right: right objects embeddings
            distance_true: true distances

        Returns:
            torch.Tensor: loss
        """
        # euclidian distance
        diff = embeddings_left - embeddings_right
        distance_pred = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))

        bs = len(distance_true)
        margin_distance = self.margin - distance_pred
        margin_distance = torch.clamp(margin_distance, min=0.0)
        loss = (1 - distance_true) * torch.pow(distance_pred, 2) + distance_true * torch.pow(
            margin_distance, 2
        )

        if self.reduction == "mean":
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class ContrastiveDistanceLoss(nn.Module):
    """The Contrastive distance loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """
        Args:
            margin: margin parameter
            reduction: criterion reduction type
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, distance_pred, distance_true) -> torch.Tensor:
        """Forward propagation method for the contrastive loss.

        Args:
            distance_pred: predicted distances
            distance_true: true distances

        Returns:
            torch.Tensor: loss
        """
        bs = len(distance_true)
        margin_distance = self.margin - distance_pred
        margin_distance = torch.clamp(margin_distance, min=0.0)
        loss = (1 - distance_true) * torch.pow(distance_pred, 2) + distance_true * torch.pow(
            margin_distance, 2
        )

        if self.reduction == "mean":
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class ContrastivePairwiseEmbeddingLoss(nn.Module):
    """ContrastivePairwiseEmbeddingLoss – proof of concept criterion.

    Still work in progress.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """
        Args:
            margin: margin parameter
            reduction: criterion reduction type
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, embeddings_pred, embeddings_true) -> torch.Tensor:
        """Forward propagation method for the contrastive loss.

        Work in progress.

        Args:
            embeddings_pred: predicted embeddings
            embeddings_true: true embeddings

        Returns:
            torch.Tensor: loss
        """
        device = embeddings_pred.device
        # s - state space
        # d - embeddings space
        # a - action space
        pairwise_similarity = torch.einsum("se,ae->sa", embeddings_pred, embeddings_true)
        bs = embeddings_pred.shape[0]
        batch_idx = torch.arange(bs, device=device)
        loss = F.cross_entropy(pairwise_similarity, batch_idx, reduction=self.reduction)
        return loss


__all__ = [
    "ContrastiveEmbeddingLoss",
    "ContrastiveDistanceLoss",
    "ContrastivePairwiseEmbeddingLoss",
]
