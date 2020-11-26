"""
MAP metric.
"""
from typing import List

import torch

from catalyst.metrics.functional import process_recsys_components


def avg_precision_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, k=10
) -> torch.Tensor:
    """
    Calculate the Average Precision for RecSys.
    The precision metric summarizes the fraction of relevant items
    out of the whole the recommendation list.

    To compute the precision at k set the threshold rank k,
    compute the percentage of relevant items in topK,
    ignoring the documents ranked lower than k.

    The average precision at k (AP at k) summarizes the average
    precision for relevant items up to the k-th one.
    Wikipedia entry for the Average precision

    <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
    oldid=793358396#Average_precision>

    If a relevant document never gets retrieved,
    we assume the precision corresponding to that
    relevant doc to be zero

    Args:
        outputs (torch.Tensor):
            Tensor weith predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        k (int):
            The position to compute the truncated AP,
            must be positive

    Returns:
        ap_score (torch.Tensor):
            The map score for each batch.
            size: [batch_size, 1]
    """
    k = min(outputs.size(1), k)
    targets_sort_by_outputs_at_k = process_recsys_components(
        outputs, targets, k
    )
    precisions = torch.zeros_like(targets_sort_by_outputs_at_k)

    for index in range(k):
        precisions[:, index] = torch.sum(
            targets_sort_by_outputs_at_k[:, : (index + 1)], dim=1
        ) / float(index + 1)

    only_relevant_precision = precisions * targets_sort_by_outputs_at_k
    ap_score = only_relevant_precision.sum(dim=1) / (
        (only_relevant_precision != 0).sum(dim=1)
    )
    ap_score[torch.isnan(ap_score)] = 0
    return ap_score


def mean_avg_precision(
    outputs: torch.Tensor, targets: torch.Tensor, topk: List[int]
) -> List[torch.Tensor]:
    """
    Calculate the mean average precision (MAP) for RecSys.
    The metrics calculate the mean of the AP across all batches

    MAP amplifies the interest in finding many
    relevant items for each query

    Args:
        outputs (torch.Tensor):
            Tensor weith predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        top_k (List[int]):
            List of parameter for evaluation
            topK items

    Returns:
        map_at_k (Tuple[float]):
            The map score for every k.
            size: len(top_k)
    """
    result = []
    for k in topk:
        result.append(torch.mean(avg_precision_at_k(outputs, targets, k)))

    return result


__all__ = ["mean_avg_precision", "avg_precision"]
