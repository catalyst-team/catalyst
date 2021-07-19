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


class BarlowTwinsLoss(nn.Module):
    """The Contrastive embedding loss.

    It has been proposed in `Barlow Twins:
    Self-Supervised Learning via Redundancy Reduction`_.

    .. _Barlow Twins: Self-Supervised Learning via Redundancy Reduction:
        https://arxiv.org/abs/2103.03230
    """

    def __init__(self, lmbda=1.0):
        """
        Args:
            lmbda: trade-off parameter
        """
        super().__init__()
        self.lmbda = lmbda

    def forward(
        self, embeddings_left: torch.Tensor, embeddings_right: torch.Tensor,
    ) -> torch.Tensor:
        """Forward propagation method for the contrastive loss.

        Args:
            embeddings_left: left objects embeddings [batch_size, features_dim]
            embeddings_right: right objects embeddings [batch_size, features_dim]

        Returns:
            torch.Tensor: loss
        """
        # normalization
        z_left = (embeddings_left - embeddings_left.mean(dim=0)) / embeddings_left.std(dim=0)
        z_right = (embeddings_right - embeddings_right.mean(dim=0)) / embeddings_right.std(dim=0)

        # cross-correlation matrix
        batch_size = z_left.shape[0]
        feature_dim = z_right.shape[1]
        cross_correlation = torch.matmul(z_left.T, z_right) / batch_size

        # selection of diagonal elements and off diagonal elements
        on_diag = torch.diagonal(cross_correlation)
        off_diag = (
            cross_correlation.flatten()[:-1]
            .view(feature_dim - 1, feature_dim + 1)[:, 1:]
            .flatten()
        )

        # the loss described in the original Barlow Twin's paper
        # encouraging off_diag to be zero and on_diag to be one
        on_diag_loss = on_diag.add_(-1).pow_(2).sum()
        off_diag_loss = off_diag.pow_(2).sum()
        loss = on_diag_loss + self.lmbda * off_diag_loss
        return loss


__all__ = [
    "ContrastiveEmbeddingLoss",
    "ContrastiveDistanceLoss",
    "ContrastivePairwiseEmbeddingLoss",
    "BarlowTwinsLoss",
]
