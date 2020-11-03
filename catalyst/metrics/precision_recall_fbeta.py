from typing import Optional, Tuple

import torch
from catalyst.metrics.functional import get_multiclass_statistics


def precision_recall_fbeta(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-6,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tn, fp, fn, tp, support = get_multiclass_statistics(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        num_classes=num_classes
    )
    precision = (tp + eps) / (fp + tp + eps)
    recall = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta**2) * precision * recall
    denominator = beta ** 2 * precision + recall
    fbeta = numerator / denominator

    return precision, recall, fbeta
