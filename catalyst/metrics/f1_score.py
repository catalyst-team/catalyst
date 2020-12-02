"""
F1 score.
"""
from typing import Optional, Union

import torch

from catalyst.metrics.classification import precision_recall_fbeta_support
from catalyst.utils.torch import get_activation_fn


def fbeta_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """
    Counts fbeta score for given ``outputs`` and ``targets``.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Raises:
        Exception: If ``beta`` is a negative number.

    Returns:
        float: F_1 score.
    """
    if beta < 0:
        raise Exception("beta parameter should be non-negative")

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
    beta: float = 1.0,
    eps: float = 1e-7,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
    threshold: float = None,
    activation: str = "Sigmoid",
) -> Union[float, torch.Tensor]:
    """
    Fbeta_score with beta=1.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known
        threshold: threshold for outputs binarization
        activation: An torch.nn activation applied to the outputs.
            Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax2d'``

    Returns:
        float: F_1 score
    """
    activation_fn = get_activation_fn(activation)

    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    score = fbeta_score(
        outputs=outputs,
        targets=targets,
        beta=beta,
        eps=eps,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )

    return score


__all__ = ["f1_score", "fbeta_score"]
