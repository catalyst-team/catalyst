from typing import Optional, Tuple

import torch

from catalyst.metrics.functional.misc import get_multiclass_statistics


def precision(tp: int, fp: int, eps: float = 1e-5, zero_division: int = 0) -> float:
    """Calculates precision (a.k.a. positive predictive value) for binary
    classification and segmentation.

    Args:
        tp: number of true positives
        fp: number of false positives
        eps: epsilon to use
        zero_division: int value, should be one of 0 or 1; if both tp==0 and fp==0 return this
            value as s result

    Returns:
        precision value (0-1)
    """
    # originally precision is: ppv = tp / (tp + fp + eps)
    # but when both masks are empty this gives: tp=0 and fp=0 => ppv=0
    # so here precision is defined as ppv := 1 - fdr (false discovery rate)
    if tp == 0 and fp == 0:
        return zero_division
    return 1 - fp / (tp + fp + eps)


def recall(tp: int, fn: int, eps=1e-5, zero_division: int = 0) -> float:
    """Calculates recall (a.k.a. true positive rate) for binary classification and segmentation.

    Args:
        tp: number of true positives
        fn: number of false negatives
        eps: epsilon to use
        zero_division: int value, should be one of 0 or 1; if both tp==0 and fn==0 return this
            value as s result

    Returns:
        recall value (0-1)
    """
    # originally recall is: tpr := tp / (tp + fn + eps)
    # but when both masks are empty this gives: tp=0 and fn=0 => tpr=0
    # so here recall is defined as tpr := 1 - fnr (false negative rate)
    if tp == 0 and fn == 0:
        return zero_division
    return 1 - fn / (fn + tp + eps)


def f1score(precision_value, recall_value, eps=1e-5):
    """Calculating F1-score from precision and recall to reduce computation redundancy.

    Args:
        precision_value: precision (0-1)
        recall_value: recall (0-1)
        eps: epsilon to use

    Returns:
        F1 score (0-1)
    """
    numerator = 2 * (precision_value * recall_value)
    denominator = precision_value + recall_value + eps
    return numerator / denominator


def precision_recall_fbeta_support(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-6,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
    zero_division: int = 0,
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
        zero_division: int value, should be one of 0 or 1;
            used for precision and recall computation

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
    # @TODO: sync between metrics
    # precision = _precision(tp=tp, fp=fp, eps=eps, zero_division=zero_division)
    precision = (tp + eps) / (fp + tp + eps)
    recall = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta ** 2) * precision * recall
    denominator = beta ** 2 * precision + recall
    fbeta = numerator / denominator

    return precision, recall, fbeta, support


__all__ = [
    "precision_recall_fbeta_support",
]
