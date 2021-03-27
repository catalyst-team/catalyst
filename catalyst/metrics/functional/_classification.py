from typing import Optional, Tuple

import numpy as np
import torch

from catalyst.metrics.functional._misc import get_multiclass_statistics


def precision(tp: int, fp: int, zero_division: int = 0) -> float:
    """Calculates precision (a.k.a. positive predictive value) for binary
    classification and segmentation.

    Args:
        tp: number of true positives
        fp: number of false positives
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
    return 1 - fp / (tp + fp)


def recall(tp: int, fn: int, zero_division: int = 0) -> float:
    """Calculates recall (a.k.a. true positive rate) for binary classification and segmentation.

    Args:
        tp: number of true positives
        fn: number of false negatives
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
    return 1 - fn / (fn + tp)


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
    Counts precision_val, recall, fbeta_score.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known.
        zero_division: int value, should be one of 0 or 1;
            used for precision_val and recall computation

    Returns:
        tuple of precision_val, recall, fbeta_score

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
            tensor([1., 1., 1.]),  # precision_val per class
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
    # precision_val = _precision(tp=tp, fp=fp, eps=eps, zero_division=zero_division)
    precision_val = (tp + eps) / (fp + tp + eps)
    recall_val = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta ** 2) * precision_val * recall_val
    denominator = beta ** 2 * precision_val + recall_val
    fbeta = numerator / denominator

    return precision_val, recall_val, fbeta, support


def get_aggregated_metrics(
    tp: np.array, fp: np.array, fn: np.array, support: np.array, zero_division: int = 0
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Count precision, recall, f1 scores per-class and with macro, weighted and micro average
    with statistics.

    Args:
        tp: array of shape (num_classes, ) of true positive statistics per class
        fp: array of shape (num_classes, ) of false positive statistics per class
        fn: array of shape (num_classes, ) of false negative statistics per class
        support: array of shape (num_classes, ) of samples count per class
        zero_division: int value, should be one of 0 or 1;
            used for precision and recall computation

    Returns:
        arrays of metrics: per-class, micro, macro, weighted averaging
    """
    num_classes = len(tp)
    precision_values = np.zeros(shape=(num_classes,))
    recall_values = np.zeros(shape=(num_classes,))
    f1_values = np.zeros(shape=(num_classes,))

    for i in range(num_classes):
        precision_values[i] = precision(tp=tp[i], fp=fp[i], zero_division=zero_division)
        recall_values[i] = recall(tp=tp[i], fn=fn[i], zero_division=zero_division)
        f1_values[i] = f1score(precision_value=precision_values[i], recall_value=recall_values[i])

    per_class = (
        precision_values,
        recall_values,
        f1_values,
        support,
    )

    macro = (
        precision_values.mean(),
        recall_values.mean(),
        f1_values.mean(),
        None,
    )

    weight = support / support.sum()
    weighted = (
        (precision_values * weight).sum(),
        (recall_values * weight).sum(),
        (f1_values * weight).sum(),
        None,
    )

    micro_precision = precision(tp=tp.sum(), fp=fp.sum(), zero_division=zero_division)
    micro_recall = recall(tp=tp.sum(), fn=fn.sum(), zero_division=zero_division)
    micro = (
        micro_precision,
        micro_recall,
        f1score(precision_value=micro_precision, recall_value=micro_recall),
        None,
    )
    return per_class, micro, macro, weighted


def get_binary_metrics(
    tp: int, fp: int, fn: int, zero_division: int
) -> Tuple[float, float, float]:
    """
    Get precision, recall, f1 score metrics from true positive, false positive,
        false negative statistics for binary classification


    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        zero_division: int value, should be 0 or 1

    Returns:
        precision, recall, f1 scores
    """
    precision_value = precision(tp=tp, fp=fp, zero_division=zero_division)
    recall_value = recall(tp=tp, fn=fn, zero_division=zero_division)
    f1_value = f1score(precision_value=precision_value, recall_value=recall_value)
    return precision_value, recall_value, f1_value


__all__ = [
    "f1score",
    "precision_recall_fbeta_support",
    "precision",
    "recall",
    "get_aggregated_metrics",
    "get_binary_metrics",
]
