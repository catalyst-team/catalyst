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
    """
    tn, fp, fn, tp, support = get_multiclass_statistics(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )
    precision = (tp + eps) / (fp + tp + eps)
    recall = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta ** 2) * precision * recall
    denominator = beta ** 2 * precision + recall
    fbeta = numerator / denominator

    return precision, recall, fbeta, support
