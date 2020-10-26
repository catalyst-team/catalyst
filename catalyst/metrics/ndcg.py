"""
Discounted Cumulative Gain metrics:
    * :func:`dcg`
    * :func:`ndcg`
"""
from typing import Sequence

import torch


def dcg(
    outputs: torch.Tensor, targets: torch.Tensor, 
    gain_function=lambda x: torch.pow(2, x) - 1, k=100
) -> torch.Tensor:
    """
    Computes DCG@topk for the specified values of `topk`.
    Reference:
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_length]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_length]
        gain_function: 
            callable, gain function for the ground truth labels. 
            on the deafult, the torch.pow(2, x) - 1 function used
            to get emphasise on the retirvng the revelant documnets.
        k (int):
            Parameter fro evaluation on top-k items

    Returns:
        list with computed dcg@topk
    """
    k = min(outputs.size(1), k)
    order = torch.argsort(outputs, descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(
        targets, dim=-1, index=order
    )

    gains = gain_function(true_sorted_by_preds)
    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0))
    discounted_gains = (gains*discounts)[:, :k]
    sum_dcg =  torch.sum(discounted_gains, dim=1)
    return sum_dcg


def ndcg(
    outputs: torch.Tensor, targets: torch.Tensor, 
    gain_function=lambda x: torch.pow(2, x) - 1, k=100
) -> torch.Tensor:
    """
    Computes nDCG@topk for the specified values of `topk`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [batch_size; slate_size]
        targets (torch.Tensor): ground truth, labels
            with shape [batch_size; slate_size]
        gain_function: 
            callable, gain function for the ground truth labels. 
            on the deafult, the torch.pow(2, x) - 1 function used
            to get emphasise on the retirvng the revelant documnets.
        k (int):
            Parameter fro evaluation on top-k items

    Returns:
        list with computed ndcg@topk
    """
    ideal_dcgs = dcg(targets, targets, gain_function, k)
    ndcg = dcg(outputs, targets, gain_function, k) / ideal_dcgs

    idcg_mask = ideal_dcgs == 0
    ndcg[idcg_mask] = 0.

    return ndcg


__all__ = ["dcg", "ndcg"]
