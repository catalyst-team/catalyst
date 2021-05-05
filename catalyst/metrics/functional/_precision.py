from typing import Optional, Union

import torch

from catalyst.metrics.functional._classification import precision_recall_fbeta_support


def precision(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    argmax_dim: int = -1,
    eps: float = 1e-7,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """
    Multiclass precision score.

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
        Tensor: precision for every class

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics
        metrics.precision(
            outputs=torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            targets=torch.tensor([0, 1, 2]),
        )
        # tensor([1., 1., 1.])


    .. code-block:: python

        import torch
        from catalyst import metrics
        metrics.precision(
            outputs=torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),
            targets=torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),
        )
        # tensor([0.5000, 0.5000]
    """
    precision_score, _, _, _, = precision_recall_fbeta_support(
        outputs=outputs, targets=targets, argmax_dim=argmax_dim, eps=eps, num_classes=num_classes,
    )
    return precision_score


__all__ = ["precision"]
