from typing import Optional, Union

import torch

from catalyst.metrics import precision_recall_fbeta_support


def recall(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    argmax_dim: int = -1,
    eps: float = 1e-7,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """
    Multiclass precision metric.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., (num_classes or 1)]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        eps: float. Epsilon to avoid zero division.
        num_classes: int, that specifies number of classes if it known.

    Returns:
        Tensor: recall for every class
    """
    _, recall_score, _, _ = precision_recall_fbeta_support(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        eps=eps,
        num_classes=num_classes,
    )

    return recall_score


__all__ = ["recall"]
