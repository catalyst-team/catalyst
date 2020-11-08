"""
MAP metric.
"""
from typing import Dict, List

import torch


def avg_precision(
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
        result (torch.Tensor):
            The map score for each batch.
            size: [batch_size, 1]
    """
    k = min(outputs.size(1), k)
    _, indices_for_sort = outputs.sort(descending=True, dim=-1)
    targets_sorted_by_outputs = torch.gather(
        targets, dim=-1, index=indices_for_sort
    )

    targets_sorted_by_outputs_topk = targets_sorted_by_outputs[:, :k]
    precisions = torch.zeros_like(targets_sorted_by_outputs_topk)

    for index in range(k):
        precisions[:, index] = torch.sum(
            targets_sorted_by_outputs_topk[:, : (index + 1)], dim=1
        ) / float(index + 1)

    only_relevant_precision = precisions * targets_sorted_by_outputs_topk
    ap = only_relevant_precision.sum(dim=1) / (
        (only_relevant_precision != 0).sum(dim=1)
    )
    ap[torch.isnan(ap)] = 0
    return ap


def mean_avg_precision(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: List[int]
) -> Dict[str, int]:
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
        result (Dict[str, int]):
            The map score for every k.
            size: [len(top_k), 1]
    """
    map_k_tuple = tuple(
        torch.mean(avg_precision(outputs, targets, k)).item() for k in top_k
    )
    return map_k_tuple


__all__ = ["mean_avg_precision", "avg_precision"]
