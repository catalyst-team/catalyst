# flake8: noqa
# TODO: refactor and add docs, move to utils
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

_EPS = 1e-8


def euclidean_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """@TODO: Docs. Contribution is welcome."""
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    if y is not None:
        y_norm = (y ** 2).sum(1).unsqueeze(0)
    else:
        y = x
        y_norm = x_norm.t()

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    dist.clamp_min_(0.0)
    return dist


def cosine_distance(x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate cosine distance between x and z.
    @TODO: Docs. Contribution is welcome.
    """
    x = F.normalize(x)

    if z is not None:
        z = F.normalize(z)
    else:
        z = x.clone()

    return torch.sub(1, torch.mm(x, z.transpose(0, 1)))


def batch_all(labels: torch.Tensor, exclude_negatives: bool = True) -> torch.Tensor:
    """Create a 3D mask of all possible triplets.
    @TODO: Docs. Contribution is welcome.
    """
    batch_size = labels.size(0)
    indices_equal = torch.eye(batch_size, device=labels.device).type(torch.bool)
    indices_not_equal = ~indices_equal

    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    yi_equal_yj = label_equal.unsqueeze(2)
    yi_equal_yk = label_equal.unsqueeze(1)

    yi_not_equal_yk = ~yi_equal_yk
    valid_labels = yi_equal_yj & yi_not_equal_yk

    mask = distinct_indices & valid_labels

    if exclude_negatives:
        mask = mask & create_negative_mask(labels)

    return mask.float()


def create_negative_mask(labels: torch.Tensor, neg_label: int = -1) -> torch.Tensor:
    """@TODO: Docs. Contribution is welcome."""
    neg_labels = torch.ge(labels, neg_label)
    pos_labels = ~neg_labels

    i_less_neg = pos_labels.unsqueeze(1).unsqueeze(2)
    j_less_neg = pos_labels.unsqueeze(1).unsqueeze(0)
    k_less_neg = pos_labels.unsqueeze(0).unsqueeze(0)

    anchors = labels.unsqueeze(1).unsqueeze(2)
    negatives = labels.unsqueeze(0).unsqueeze(0)
    k_equal = torch.eq(anchors + neg_label, negatives)

    k_less_or_equal = k_equal | k_less_neg
    mask = i_less_neg & j_less_neg & k_less_or_equal

    return mask


def triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """@TODO: Docs. Contribution is welcome."""
    cosine_dists = cosine_distance(embeddings)
    mask = batch_all(labels)

    anchor_positive_dist = cosine_dists.unsqueeze(2)
    anchor_negative_dist = cosine_dists.unsqueeze(1)
    triplet_loss_value = F.relu(anchor_positive_dist - anchor_negative_dist + margin)
    triplet_loss_value = torch.mul(triplet_loss_value, mask)

    num_positive_triplets = torch.gt(triplet_loss_value, _EPS).sum().float()
    triplet_loss_value = triplet_loss_value.sum() / (num_positive_triplets + _EPS)

    return triplet_loss_value


def _create_margin_mask(labels: torch.Tensor) -> torch.Tensor:
    equal_labels_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    marign_mask = 2 * equal_labels_mask.float() - 1
    return marign_mask


def _skip_labels_mask(labels: torch.Tensor, skip_labels: Union[int, List[int]]) -> torch.Tensor:
    skip_labels = torch.tensor(skip_labels, dtype=labels.dtype, device=labels.device).reshape(-1)
    skip_condition = (labels.unsqueeze(-1) == skip_labels).any(-1)
    skip_mask = ~(skip_condition.unsqueeze(-1) & skip_condition.unsqueeze(0))
    return skip_mask


def margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 1.0,
    skip_labels: Union[int, List[int]] = -1,
) -> torch.Tensor:
    """@TODO: Docs. Contribution is welcome."""
    embeddings = F.normalize(embeddings, p=2.0, dim=1)
    distances = euclidean_distance(embeddings, embeddings)

    margin_mask = _create_margin_mask(labels)
    skip_mask = _skip_labels_mask(labels, skip_labels).float()
    loss = torch.mul(
        skip_mask, F.relu(alpha + torch.mul(margin_mask, torch.sub(distances, beta))),
    )
    return loss.sum() / (skip_mask.sum() + _EPS)
