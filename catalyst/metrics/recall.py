from typing import Optional

import torch

from catalyst.metrics.functional import get_multiclass_statistics


def recall(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    argmax_dim: int = -1,
    eps: float = 1e-7,
    num_classes: Optional[int] = None,

) -> torch.Tensor:
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
        num_classes: int, that specifies number of classes if it known

    Returns:
        Tensor: recall for every class
    """
    _, _, fn, tp, _ = get_multiclass_statistics(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )

    return (tp + eps)/(fn + tp + eps)
