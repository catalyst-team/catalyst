"""
F1 score.
"""
from typing import Optional

import torch

from catalyst.metrics.functional import get_multiclass_statistics


def fbeta_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Returns:
        float: F_1 score
     """
    if beta < 0:
        raise Exception("beta parameter should be non-negative")

    _, fp, fn, tp, _ = get_multiclass_statistics(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )

    precision_plus_recall = (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps

    score = ((1 + beta ** 2) * tp + eps) / precision_plus_recall
    return score


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> float:
    """
    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Returns:
        float: F_1 score
    """
    score = fbeta_score(
        outputs=outputs,
        targets=targets,
        beta=1,
        eps=eps,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )

    return score


__all__ = ["f1_score", "fbeta_score"]
