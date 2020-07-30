"""
Discounted Cumulative Gain metrics:
    * :func:`dcg`
    * :func:`ndcg`
"""
from typing import Sequence

import torch


def dcg(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (10,)
) -> Sequence[torch.Tensor]:
    """
    Computes DCG@topk for the specified values of `topk`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_length]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_length]
        topk (Sequence[int]): `topk` for dcg@topk computing

    Returns:
        list with computed dcg@topk
    """
    order = torch.argsort(outputs, descending=True, dim=-1)

    dcg_scores = []
    for k in topk:
        if len(outputs.shape) == 1:
            targets_at_k = torch.take(targets, order[:k])
        else:
            targets_at_k = torch.take(
                targets, order.narrow(-1, 0, min(k, order.shape[1]))
            )

        discounts_at_k = torch.log2(torch.arange(targets_at_k.shape[-1]) + 2.0)
        dcg_scores.append(torch.sum(targets_at_k / discounts_at_k).mean())

    return dcg_scores


def ndcg(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (10,),
) -> Sequence[torch.Tensor]:
    """
    Computes nDCG@topk for the specified values of `topk`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_size]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_size]
        topk (Sequence[int]): `topk` for ndcg@topk computing

    Returns:
        list with computed ndcg@topk
    """
    ideal_dcgs = dcg(targets, targets, topk)
    actual_dcgs = dcg(outputs, targets, topk)

    ndcg_scores = []
    for actual, ideal in zip(ideal_dcgs, actual_dcgs):
        if ideal != 0:
            ndcg_scores.append(actual / ideal)
        else:
            ndcg_scores.append(torch.tensor(0.0))
    return ndcg_scores


__all__ = ["dcg", "ndcg"]
