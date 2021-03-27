from typing import List

import torch

from catalyst.metrics.functional._misc import process_recsys_components


def reciprocal_rank(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculate the Reciprocal Rank (MRR)
    score given model outputs and targets
    Data aggregated in batches.

    Args:
        outputs:
            Tensor weith predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets:
            Binary tensor with ground truth.
            1 means the item is relevant
            and 0 if it's not relevant
            size: [batch_size, slate_length]
            ground truth, labels
        k:
            Parameter for evaluation on top-k items

    Returns:
        MRR score

    Examples:
        >>> reciprocal_rank(
        >>>     outputs=torch.Tensor([
        >>>         [4.0, 2.0, 3.0, 1.0],
        >>>         [1.0, 2.0, 3.0, 4.0],
        >>>     ]),
        >>>     targets=torch.Tensor([
        >>>         [0, 0, 1.0, 1.0],
        >>>         [0, 0, 1.0, 1.0],
        >>>     ]),
        >>>     k=1,
        >>> )
        tensor([[0.], [1.]])
        >>> reciprocal_rank(
        >>>     outputs=torch.Tensor([
        >>>         [4.0, 2.0, 3.0, 1.0],
        >>>         [1.0, 2.0, 3.0, 4.0],
        >>>     ]),
        >>>     targets=torch.Tensor([
        >>>         [0, 0, 1.0, 1.0],
        >>>         [0, 0, 1.0, 1.0],
        >>>     ]),
        >>>     k=3,
        >>> )
        tensor([[0.5000], [1.0000]])
    """
    k = min(outputs.size(1), k)
    targets_sort_by_outputs_at_k = process_recsys_components(outputs, targets)[:, :k]
    values, indices = torch.max(targets_sort_by_outputs_at_k, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t()
    mrr_score = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = values == 0.0
    mrr_score[zero_sum_mask] = 0.0
    return mrr_score


def mrr(outputs: torch.Tensor, targets: torch.Tensor, topk: List[int]) -> List[torch.Tensor]:
    """
    Calculate the Mean Reciprocal Rank (MRR)
    score given model outputs and targets
    Data aggregated in batches.

    The MRR@k is the mean overall batch of the
    reciprocal rank, that is the rank of the highest
    ranked relevant item, if any in the top *k*, 0 otherwise.
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    Args:
        outputs:
            Tensor weith predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets:
            Binary tensor with ground truth.
            1 means the item is relevant
            and 0 if it's not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        topk:
            Parameter fro evaluation on top-k items

    Returns:
        MRR score

    Examples:
        >>> mrr(
        >>>     outputs=torch.Tensor([
        >>>         [4.0, 2.0, 3.0, 1.0],
        >>>         [1.0, 2.0, 3.0, 4.0],
        >>>     ]),
        >>>     targets=torch.Tensor([
        >>>         [0, 0, 1.0, 1.0],
        >>>         [0, 0, 1.0, 1.0],
        >>>     ]),
        >>>     topk=[1, 3],
        >>> )
        [tensor(0.5000), tensor(0.7500)]
    """
    results = []
    for k in topk:
        results.append(torch.mean(reciprocal_rank(outputs, targets, k)))

    return results


__all__ = ["reciprocal_rank", "mrr"]
