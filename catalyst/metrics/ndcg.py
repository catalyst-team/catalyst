"""
Discounted Cumulative Gain metrics
"""
from typing import List

import torch

from catalyst.metrics.functional import process_recsys


def dcg(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k=10,
    gain_function="pow_rank",
) -> torch.Tensor:
    """
    Computes DCG@topk for the specified values of `k`.
    Graded relevance as a measure of  usefulness,
    or gain, from examining a set of items.
    Gain may be reduced at lower ranks.
    Reference:
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_length]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_length]
        gain_function:
            String indicates the gain function for the ground truth labels.
            Two options available:
            - `pow_rank`: torch.pow(2, x) - 1
            - `rank`: x
            On the default, `pow_rank` is used
            to emphasize on retrieving the relevant documents.
        k (int):
            Parameter fro evaluation on top-k items

    Returns:
        dcg_score (torch.Tensor):
            The dcg score at k

    Raises:
        ValueError: gain function can be either `pow_rank` or `rank`
    """
    k = min(outputs.size(1), k)
    targets_sorted_by_outputs_at_k = process_recsys(outputs, targets, k)

    if gain_function == "pow_rank":
        gain_function = lambda x: torch.pow(2, x) - 1
        gains = gain_function(targets_sorted_by_outputs_at_k)
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(targets_sorted_by_outputs_at_k.shape[1], dtype=torch.float)
            + 2.0
        )
        discounted_gains = (gains * discounts)[:, :k]

    elif gain_function == "rank":
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(targets_sorted_by_outputs_at_k.shape[1], dtype=torch.float)
            + 1.0
        )
        discounts[0] = 1
        discounted_gains = (targets_sorted_by_outputs_at_k * discounts)[:, :k]

    else:
        raise ValueError("gain function can be either pow_rank or rank")

    dcg_score = torch.sum(discounted_gains, dim=1)
    return dcg_score


def ndcg(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    top_k: List[int],
    gain_function="pow_rank",
) -> torch.Tensor:
    """
    Computes nDCG@topk for the specified values of `top_k`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_size]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_size]
        gain_function:
            callable, gain function for the ground truth labels.
            on the deafult, the torch.pow(2, x) - 1 function used
            to get emphasise on the retirvng the revelant documnets.
        top_k (List[int]):
            Parameter fro evaluation on top-k items

    Returns:
        tuple with computed ndcg@topk
    """
    ndcg_k_tuple = ()
    for k in top_k:
        ideal_dcgs = dcg(targets, targets, k, gain_function)
        predicted_dcgs = dcg(outputs, targets, k, gain_function)
        ndcg_score = predicted_dcgs / ideal_dcgs
        idcg_mask = ideal_dcgs == 0
        ndcg_score[idcg_mask] = 0.0
        ndcg_k_tuple += (ndcg_score,)

    return ndcg_k_tuple


__all__ = ["dcg", "ndcg"]
