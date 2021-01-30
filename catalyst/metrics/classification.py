from typing import Optional, Tuple

import torch

from catalyst.metrics.functional import get_multiclass_statistics


def precision_recall_fbeta_support(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-6,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Counts precision, recall, fbeta_score.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known.

    Returns:
        tuple of precision, recall, fbeta_score

    Examples:
        >>> precision_recall_fbeta_support(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>>     beta=1,
        >>> )
        (
            tensor([1., 1., 1.]),  # precision per class
            tensor([1., 1., 1.]),  # recall per class
            tensor([1., 1., 1.]),  # fbeta per class
            tensor([1., 1., 1.]),  # support per class
        )
        >>> precision_recall_fbeta_support(
        >>>     outputs=torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),
        >>>     targets=torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),
        >>>     beta=1,
        >>> )
        (
            tensor([0.5000, 0.5000]),  # precision per class
            tensor([0.5000, 0.5000]),  # recall per class
            tensor([0.5000, 0.5000]),  # fbeta per class
            tensor([4., 4.]),          # support per class
        )
    """
    tn, fp, fn, tp, support = get_multiclass_statistics(
        outputs=outputs, targets=targets, argmax_dim=argmax_dim, num_classes=num_classes,
    )
    precision = (tp + eps) / (fp + tp + eps)
    recall = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta ** 2) * precision * recall
    denominator = beta ** 2 * precision + recall
    fbeta = numerator / denominator

    return precision, recall, fbeta, support
