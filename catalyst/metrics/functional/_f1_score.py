from typing import Optional, Union

import torch

from catalyst.metrics.functional._classification import precision_recall_fbeta_support


def fbeta_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """Counts fbeta score for given ``outputs`` and ``targets``.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Raises:
        ValueError: If ``beta`` is a negative number.

    Returns:
        float: F_1 score.
    """
    if beta < 0:
        raise ValueError("beta parameter should be non-negative")

    _p, _r, fbeta, _ = precision_recall_fbeta_support(
        outputs=outputs,
        targets=targets,
        beta=beta,
        eps=eps,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )
    return fbeta


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """Fbeta_score with beta=1.

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
