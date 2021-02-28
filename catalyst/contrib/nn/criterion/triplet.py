# flake8: noqa
from typing import List, TYPE_CHECKING, Union

import torch
from torch import nn, Tensor
from torch.nn import TripletMarginLoss

from catalyst.contrib.nn.criterion.functional import triplet_loss
from catalyst.utils.misc import convert_labels2list

if TYPE_CHECKING:
    from catalyst.data.sampler_inbatch import IInbatchTripletSampler

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Adapted from: https://github.com/NegatioN/OnlineMiningTripletLoss
    """

    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: margin for triplet
        """
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: if true, output is the pairwise squared euclidean
                distance matrix. If false, output is the pairwise euclidean
                distance matrix

        Returns:
            torch.Tensor: pairwise matrix of size (batch_size, batch_size)
        """
        # Get squared L2 norm for each embedding.
        # We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability
        # (the diagonal of the result will be exactly 0).
        # shape (batch_size)
        square = torch.mm(embeddings, embeddings.t())
        diag = torch.diag(square)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = diag.view(-1, 1) - 2.0 * square + diag.view(1, -1)

        # Because of computation errors, some distances
        # might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite
            # when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, p] is True
        if a and p are distinct and have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            torch.Tensor: mask with shape [batch_size, batch_size]
        """
        indices_equal = torch.eye(labels.size(0)).type(torch.bool)

        # labels and indices should be on
        # the same device, otherwise - exception
        indices_equal = indices_equal.to("cuda" if labels.is_cuda else "cpu")

        # Check that i and j are distinct

        indices_equal = indices_equal.type(TORCH_BOOL)
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument
        # has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    def _get_anchor_negative_triplet_mask(self, labels):
        """Return 2D mask where mask[a, n] is True if a and n have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            torch.Tensor: mask with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument
        # has shape (1, batch_size) and the 2nd (batch_size, 1)
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

    def _batch_hard_triplet_loss(
        self, embeddings, labels, margin, squared=True,
    ):
        """
        Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and
        hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared
                     euclidean distance matrix. If false, output is the
                     pairwise euclidean distance matrix.

        Returns:
            torch.Tensor: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid
        # positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid
        # (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative
        # (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row
        # to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )

        # shape (batch_size)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + margin
        tl[tl < 0] = 0
        loss = tl.mean()

        return loss

    def forward(self, embeddings, targets):
        """Forward propagation method for the triplet loss.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            targets: labels of the batch, of size (batch_size)

        Returns:
            torch.Tensor: scalar tensor containing the triplet loss
        """
        return self._batch_hard_triplet_loss(embeddings, targets, self.margin)


class TripletLossV2(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, margin=0.3):
        """
        Args:
            margin: margin for triplet.
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, targets):
        """@TODO: Docs. Contribution is welcome."""
        return triplet_loss(embeddings, targets, margin=self.margin)


class TripletPairwiseEmbeddingLoss(nn.Module):
    """TripletPairwiseEmbeddingLoss – proof of concept criterion.

    Still work in progress.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(self, margin: float = 0.3, reduction: str = "mean"):
        """
        Args:
            margin: margin parameter
            reduction: criterion reduction type
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or "none"

    def forward(self, embeddings_pred, embeddings_true):
        """
        Work in progress.

        Args:
            embeddings_pred: predicted embeddings
                with shape [batch_size, embedding_size]
            embeddings_true: true embeddings
                with shape [batch_size, embedding_size]

        Returns:
            torch.Tensor: loss
        """
        device = embeddings_pred.device
        # s - state space
        # d - embeddings space
        # a - action space
        # [batch_size, embedding_size] x [batch_size, embedding_size]
        # -> [batch_size, batch_size]
        pairwise_similarity = torch.einsum("se,ae->sa", embeddings_pred, embeddings_true)
        bs = embeddings_pred.shape[0]
        batch_idx = torch.arange(bs, device=device)
        negative_similarity = pairwise_similarity + torch.diag(
            torch.full([bs], -(10 ** 9), device=device)
        )
        # TODO argsort, take k worst
        hard_negative_ids = negative_similarity.argmax(dim=-1)

        negative_similarities = pairwise_similarity[batch_idx, hard_negative_ids]
        positive_similarities = pairwise_similarity[batch_idx, batch_idx]
        loss = torch.relu(self.margin - positive_similarities + negative_similarities)
        if self.reduction == "mean":
            loss = torch.sum(loss) / bs
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class TripletMarginLossWithSampler(nn.Module):
    """
    This class combines in-batch sampling of triplets and
    default TripletMargingLoss from PyTorch.
    """

    def __init__(self, margin: float, sampler_inbatch: "IInbatchTripletSampler"):
        """
        Args:
            margin: margin value
            sampler_inbatch: sampler for forming triplets inside the batch
        """
        super().__init__()
        self._sampler_inbatch = sampler_inbatch
        self._triplet_margin_loss = TripletMarginLoss(margin=margin)

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: features with shape [batch_size, features_dim]
            labels: labels of samples having batch_size elements

        Returns: loss value

        """
        labels_list = convert_labels2list(labels)

        features_anchor, features_positive, features_negative = self._sampler_inbatch.sample(
            features=features, labels=labels_list
        )

        loss = self._triplet_margin_loss(
            anchor=features_anchor, positive=features_positive, negative=features_negative,
        )
        return loss


__all__ = [
    "TripletLoss",
    "TripletLossV2",
    "TripletPairwiseEmbeddingLoss",
    "TripletMarginLossWithSampler",
]
