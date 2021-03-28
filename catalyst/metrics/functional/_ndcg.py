from typing import List

import torch

from catalyst.metrics.functional._misc import process_recsys_components


def dcg(outputs: torch.Tensor, targets: torch.Tensor, gain_function="exp_rank") -> torch.Tensor:
    """
    Computes Discounted cumulative gain (DCG)
    DCG@topk for the specified values of `k`.
    Graded relevance as a measure of  usefulness,
    or gain, from examining a set of items.
    Gain may be reduced at lower ranks.
    Reference:
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        outputs: model outputs, logits
            with shape [batch_size; slate_length]
        targets: ground truth, labels
            with shape [batch_size; slate_length]
        gain_function:
            String indicates the gain function for the ground truth labels.
            Two options available:
            - `exp_rank`: torch.pow(2, x) - 1
            - `linear_rank`: x
            On the default, `exp_rank` is used
            to emphasize on retrieving the relevant documents.

    Returns:
        dcg_score (torch.Tensor):
            The discounted gains tensor

    Raises:
        ValueError: gain function can be either `pow_rank` or `rank`

    Examples:
        >>> dcg(
        >>>     outputs = torch.tensor([
        >>>         [3, 2, 1, 0],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [2.0, 2.0, 1.0, 0.0],
        >>>     ]),
        >>>     gain_function="linear_rank",
        >>> )
        tensor([[2.0000, 2.0000, 0.6309, 0.0000]])
        >>> dcg(
        >>>     outputs = torch.tensor([
        >>>         [3, 2, 1, 0],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [2.0, 2.0, 1.0, 0.0],
        >>>     ]),
        >>>     gain_function="linear_rank",
        >>> ).sum()
        tensor(4.6309)
        >>> dcg(
        >>>     outputs = torch.tensor([
        >>>         [3, 2, 1, 0],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [2.0, 2.0, 1.0, 0.0],
        >>>     ]),
        >>>     gain_function="exp_rank",
        >>> )
        tensor([[3.0000, 1.8928, 0.5000, 0.0000]])
        >>> dcg(
        >>>     outputs = torch.tensor([
        >>>         [3, 2, 1, 0],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [2.0, 2.0, 1.0, 0.0],
        >>>     ]),
        >>>     gain_function="exp_rank",
        >>> ).sum()
        tensor(5.3928)
    """
    targets_sort_by_outputs = process_recsys_components(outputs, targets)
    target_device = targets_sort_by_outputs.device

    if gain_function == "exp_rank":
        gain_function = lambda x: torch.pow(2, x) - 1
        gains = gain_function(targets_sort_by_outputs)
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(
                targets_sort_by_outputs.shape[1], dtype=torch.float, device=target_device,
            )
            + 2.0
        )
        discounted_gains = gains * discounts

    elif gain_function == "linear_rank":
        discounts = torch.tensor(1) / torch.log2(
            torch.arange(
                targets_sort_by_outputs.shape[1], dtype=torch.float, device=target_device,
            )
            + 1.0
        )
        discounts[0] = 1
        discounted_gains = targets_sort_by_outputs * discounts

    else:
        raise ValueError("gain function can be either exp_rank or linear_rank")

    dcg_score = discounted_gains
    return dcg_score


def ndcg(
    outputs: torch.Tensor, targets: torch.Tensor, topk: List[int], gain_function="exp_rank",
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
            - `linear_rank`: x
            On the default, `exp_rank` is used
            to emphasize on retrieving the relevant documents.
        topk (List[int]):
            Parameter fro evaluation on top-k items

    Returns:
        results (Tuple[float]):
            tuple with computed ndcg@topk

    Examples:
        >>> ndcg(
        >>>     outputs = torch.tensor([
        >>>         [0.5, 0.2, 0.1],
        >>>         [0.5, 0.2, 0.1],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [1.0, 0.0, 1.0],
        >>>         [1.0, 0.0, 1.0],
        >>>     ]),
        >>>     topk=[2],
        >>>     gain_function="exp_rank",
        >>> )
        [tensor(0.6131)]
        >>> ndcg(
        >>>     outputs = torch.tensor([
        >>>         [0.5, 0.2, 0.1],
        >>>         [0.5, 0.2, 0.1],
        >>>     ]),
        >>>     targets = torch.Tensor([
        >>>         [1.0, 0.0, 1.0],
        >>>         [1.0, 0.0, 1.0],
        >>>     ]),
        >>>     topk=[2],
        >>>     gain_function="exp_rank",
        >>> )
        [tensor(0.5000)]
    """
    results = []
    for k in topk:
        ideal_dcgs = dcg(targets, targets, gain_function)[:, :k]
        predicted_dcgs = dcg(outputs, targets, gain_function)[:, :k]
        ideal_dcgs_score = torch.sum(ideal_dcgs, dim=1)
        predicted_dcgs_score = torch.sum(predicted_dcgs, dim=1)
        ndcg_score = predicted_dcgs_score / ideal_dcgs_score
        idcg_mask = ideal_dcgs_score == 0
        ndcg_score[idcg_mask] = 0.0
        results.append(torch.mean(ndcg_score))
    return results


__all__ = ["dcg", "ndcg"]
