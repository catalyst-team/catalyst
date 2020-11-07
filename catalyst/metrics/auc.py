from typing import Tuple

import numpy as np

import torch
from torch.nn import functional as F


def _binary_auc(
    scores: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Binary AUC computation.

    Args:
        scores: estimated scores from a model.
        targets:ground truth (correct) target values.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: measured roc-auc,
        true positive rate, false positive rate
    """
    targets = targets.numpy()

    # sorting the arrays
    scores, sortind = torch.sort(scores, dim=0, descending=True)
    scores = scores.numpy()
    sortind = sortind.numpy()

    # creating the roc curve
    tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
    fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

    for i in range(1, scores.size + 1):
        if targets[sortind[i - 1]] == 1:
            tpr[i] = tpr[i - 1] + 1
            fpr[i] = fpr[i - 1]
        else:
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1] + 1

    tpr /= targets.sum() * 1.0  # noqa: WPS345
    fpr /= (targets - 1.0).sum() * -1.0

    # calculating area under curve using trapezoidal rule
    n = tpr.shape[0]
    h = fpr[1:n] - fpr[: n - 1]
    sum_h = np.zeros(fpr.shape)
    sum_h[: n - 1] = h
    sum_h[1:n] += h
    area = (sum_h * tpr).sum() / 2.0

    return area, tpr, fpr


def auc(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    AUC metric.

    Args:
        outputs: [bs; num_classes] estimated scores from a model.
        targets: [bs; num_classes] ground truth (correct) target values.

    Returns:
        torch.Tensor: Tensor with [num_classes] shape of per-class-aucs
    """
    if len(outputs) == 0:
        return 0.5

    if len(outputs.shape) < 2:
        outputs.unsqueeze_(dim=1)
    if torch.max(outputs) > 1:
        outputs = torch.sigmoid(outputs)
    num_classes = outputs.shape[1]

    if len(targets.shape) < 2:
        targets = (
            F.one_hot(targets, num_classes).float()
            if num_classes > 1
            else targets.unsqueeze_(dim=1)
        )

    assert outputs.shape == targets.shape

    per_class_auc = []
    for class_i in range(outputs.shape[1]):
        per_class_auc.append(
            _binary_auc(outputs[:, class_i], targets[:, class_i])[0]
        )
    output = torch.Tensor(per_class_auc)
    return output


__all__ = ["auc"]
