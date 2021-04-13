from typing import Tuple

import numpy as np
import torch


def binary_auc(
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

    .. warning::

        This metric is under API improvement.
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


def auc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes ROC-AUC.

    Args:
        scores: NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)

    Returns:
        torch.Tensor: Tensor with [num_classes] shape of per-class-aucs

    Example:
        >>> auc(
        >>>     scores=torch.tensor([
        >>>         [0.9, 0.1],
        >>>         [0.1, 0.9],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0],
        >>>         [0, 1],
        >>>     ]),
        >>> )
        tensor([1., 1.])
        >>> auc(
        >>>     scores=torch.tensor([
        >>>         [0.9],
        >>>         [0.8],
        >>>         [0.7],
        >>>         [0.6],
        >>>         [0.5],
        >>>         [0.4],
        >>>         [0.3],
        >>>         [0.2],
        >>>         [0.1],
        >>>         [0.0],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [0],
        >>>         [1],
        >>>         [1],
        >>>         [1],
        >>>         [1],
        >>>         [1],
        >>>         [1],
        >>>         [0],
        >>>         [0],
        >>>         [0],
        >>>     ]),
        >>> )
        tensor([0.7500])

    .. warning::

        This metric is under API improvement.
    """
    if len(scores) == 0:
        return 0.5
    if len(scores.shape) < 2:
        scores.unsqueeze_(dim=1)
    if len(targets.shape) < 2:
        targets.unsqueeze_(dim=1)
    assert scores.shape == targets.shape

    per_class_auc = []
    for class_i in range(scores.shape[1]):
        per_class_auc.append(binary_auc(scores[:, class_i], targets[:, class_i])[0])
    output = torch.Tensor(per_class_auc)
    return output


__all__ = ["binary_auc", "auc"]
