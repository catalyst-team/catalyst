"""
Discounted Cumulative Gain metrics
"""
from typing import List

import torch

from catalyst.metrics.functional import process_recsys_components


def dcg(
    outputs: torch.Tensor, targets: torch.Tensor, gain_function="exp_rank",
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
            - `exp_rank`: torch.pow(2, x) - 1
            - `rank`: x
            On the default, `exp_rank` is used
            to emphasize on retrieving the relevant documents.

    Returns:
        dcg_score (torch.Tensor):
            The discounted gains tensor

    Raises:
        ValueError: gain function can be either `pow_rank` or `rank`
    """
    targets_sort_by_outputs = process_recsys_components(outputs, targets)

    if gain_function == "exp_rank":
        gain_function = lambda x: torch.pow(2, x) - 1
        gains = gain_function(targets_sort_by_outputs)
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(targets_sort_by_outputs.shape[1], dtype=torch.float)
            + 2.0
        )
        discounted_gains = gains * discounts

    elif gain_function == "linear_rank":
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(targets_sort_by_outputs.shape[1], dtype=torch.float)
            + 1.0
        )
        discounts[0] = 1
        discounted_gains = targets_sort_by_outputs * discounts

    else:
        raise ValueError("gain function can be either exp_rank or linear_rank")

    dcg_score = discounted_gains
    return dcg_score


def ndcg(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: List[int],
    gain_function="exp_rank",
) -> List[torch.Tensor]:
    """
    Computes nDCG@topk for the specified values of `topk`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_size]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_size]
        gain_function:
            callable, gain function for the ground truth labels.
            Two options available:
            - `exp_rank`: torch.pow(2, x) - 1
            - `rank`: x
            On the default, `exp_rank` is used
            to emphasize on retrieving the relevant documents.
        topk (List[int]):
            Parameter fro evaluation on top-k items

    Returns:
        result (Tuple[float]):
        tuple with computed ndcg@topk
    """

    result = []
    for k in topk:
        ideal_dcgs = dcg(targets, targets, gain_function)[:, :k]
        predicted_dcgs = dcg(outputs, targets, gain_function)[:, :k]
        ideal_dcgs_score = torch.sum(ideal_dcgs, dim=1)
        predicted_dcgs_score = torch.sum(predicted_dcgs, dim=1)
        ndcg_score = predicted_dcgs_score / ideal_dcgs_score
        idcg_mask = ideal_dcgs_score == 0
        ndcg_score[idcg_mask] = 0.0
        result.append(torch.mean(ndcg_score))
    return result


__all__ = ["dcg", "ndcg"]
