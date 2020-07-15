from typing import Tuple

import numpy as np

import torch


def average_precision(outputs, targets, k: int = 10):
    """Computes the average precision at k.

    This function computes the average
    precision at k between two lists of items.

    Args:
        outputs (list): A list of predicted elements
        targets (list):  A list of elements that are to be predicted
        k (int, optional): The maximum number of predicted elements

    Returns:
        float: The average precision at k over the input lists
    """
    if len(outputs) > k:
        outputs = outputs[:k]

    score = 0.0
    num_hits = 0.0

    for i, predict in enumerate(outputs):
        if predict in targets and predict not in outputs[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not targets:
        return 0.0

    return score / min(len(targets), k)


def mean_average_precision(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple = (1,),
):
    """Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Args:
        outputs (list): A list of lists of predicted elements
        targets (list): A list of lists of elements that are to be predicted
        topk (int, optional): The maximum number of predicted elements

    Returns:
        float: The mean average precision at k over the input lists
    """
    max_k = max(topk)
    _, pred = outputs.topk(max_k, 1, True, True)  # noqa: WPS425

    targets = targets.cpu().detach().numpy().tolist()
    actual_list = []
    for a in targets:
        actual_list.append([a])
    targets = actual_list
    pred = pred.tolist()

    res = []
    for k in topk:
        ap = np.mean(
            [average_precision(p, a, k) for a, p in zip(targets, pred)]
        )
        res.append(ap)
    return res


__all__ = ["average_precision", "mean_average_precision"]
