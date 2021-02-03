from typing import Any, Dict, Optional, Tuple, Union
from collections import defaultdict
from functools import partial

import numpy as np

import torch

from catalyst.metrics.functional.misc import (
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)
from catalyst.metrics.metric import ICallbackLoaderMetric


def f1score(precision_value, recall_value, eps=1e-5):
    """
    Calculating F1-score from precision and recall to reduce computation
    redundancy.
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


def precision(
    tp: int, fp: int, eps: float = 1e-5, zero_division: int = 1
) -> float:
    """
    Calculates precision (a.k.a. positive predictive value) for binary
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


def recall(tp: int, fn: int, eps=1e-5, zero_division: int = 1) -> float:
    """
    Calculates recall (a.k.a. true positive rate) for binary classification and
    segmentation.
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


class StatisticsMetric(ICallbackLoaderMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from data.

    It can work in binary, multiclass and multilabel classification tasks.
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        mode: str = "binary",
        num_classes: int = 2,
    ):
        """
        Init params
        Args:
            compute_on_call: if True, computes and returns metric value during metric call
            prefix:
            suffix:
            mode: one of "binary", "multiclass" and "multilabel"
            num_classes: number of classes

        Raises:
            ValueError: if mode is incorrect
        """
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix,
        )
        if mode == "binary":
            self.statistics_fn = get_binary_statistics
        elif mode == "multiclass":
            self.statistics_fn = partial(
                get_multiclass_statistics, num_classes=num_classes
            )
        elif mode == "multilabel":
            self.statistics_fn = get_multilabel_statistics
        else:
            raise ValueError(
                f'Mode should be one of "binary", "multiclass", "multilabel".'
            )

        self.num_classes = num_classes
        self.statistics = None
        super().reset(num_batches=0, num_samples=0)

    def reset(self, num_batches: int, num_samples: int) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(
            lambda: np.zeros(shape=(self.num_classes,))
        )

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compute statistics from outputs and targets, update accumulated statistics with new values.
        Args:
            outputs: prediction values
            targets: true answers
        """
        tn, fp, fn, tp, support = self.statistics_fn(
            outputs=outputs.cpu().detach(), targets=targets.cpu().detach(),
        )
        self.statistics["tn"] += tn.numpy()
        self.statistics["fp"] += fp.numpy()
        self.statistics["fn"] += fn.numpy()
        self.statistics["tp"] += tp.numpy()
        self.statistics["support"] += support.numpy()

    def compute(self) -> Dict[str, Union[int, np.array]]:
        """
        Return accumulated statistics
        Returns:
            dict of statistics
        """
        return self.statistics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Return accumulated statistics
        Returns:
            dict of statistics

        Examples:
            >>> For binary mode: {"tp": 3, "fp": 4, "tn": 5, "fn": 1, "support": 13}
            >>> For other modes: {"tp": np.array([1, 2, 1]), "fp": np.array([2, 1, 0]), ...}
        """
        result = self.compute()
        return {k: result[k] for k in sorted(result.keys())}


class PrecisionRecallF1SupportMetric(StatisticsMetric):
    """
    Metric that can collect statistics and count precision, recall, f1_score and support with it.
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        num_classes: int = 2,
        mode: str = "binary",
        zero_division: int = 0,
    ) -> None:
        """
        Init PrecisionRecallF1SupportMetric instance

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
            num_classes: number of classes in loader's dataset
            mode: one of "binary", "multiclass" and "multilabel"
            zero_division: value to set in case of zero division during metrics
                (precision, recall) computation; should be one of 0 or 1
        """
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            mode=mode,
        )
        self.zero_division = zero_division
        self.metrics = None
        self.reset(num_batches=0, num_samples=0)

    def reset(self, num_batches: int, num_samples: int) -> None:
        """Reset all the statistics and metrics fields"""
        super().reset(num_batches=num_batches, num_samples=num_samples)
        self.metrics = defaultdict(float)

    def _compute_weighted_average(
        self,
        precision_values: np.array,
        recall_values: np.array,
        f1_values: np.array,
        weights: Union[float, np.array],
    ) -> Dict[str, float]:
        """
        Compute average with weights fot precision_values, recall_values, f1_values
        Args:
            precision_values: array of per-class precision values
            recall_values: array of per-class recall values
            f1_values: array of per-class f1 values
            weights: class weights, float or array of weights

        Returns:
            dict of averaged metrics
        """
        weighted_metrics = defaultdict(float)
        for metric_name, metric_value in zip(
            ("precision", "recall", "f1"),
            (precision_values, recall_values, f1_values),
        ):
            weighted_metrics[metric_name] = (metric_value * weights).sum()
        return weighted_metrics

    def _compute_micro_average(self) -> Dict[str, float]:
        """
        Compute micro precision, recall and f1_score values with statistics.
        Returns:
            dict of averaged metrics
        """
        micro_metrics = defaultdict(float)
        micro_metrics["precision"] = (
            self.statistics["tp"].sum()
            / (self.statistics["tp"].sum() + self.statistics["fp"].sum())
        ).item()
        micro_metrics["recall"] = (
            self.statistics["tp"].sum()
            / (self.statistics["tp"].sum() + self.statistics["fn"].sum())
        ).item()
        micro_metrics["f1"] = f1score(
            precision_value=micro_metrics["precision"],
            recall_value=micro_metrics["recall"],
        )
        return micro_metrics

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        If not binary, compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics
        """
        precision_values, recall_values, f1_values = (
            np.zeros(shape=(self.num_classes,)),
            np.zeros(shape=(self.num_classes,)),
            np.zeros(shape=(self.num_classes,)),
        )

        for i in range(self.num_classes):
            precision_values[i] = precision(
                tp=self.statistics["tp"][i],
                fp=self.statistics["fp"][i],
                zero_division=self.zero_division,
            )
            recall_values[i] = recall(
                tp=self.statistics["tp"][i],
                fn=self.statistics["fn"][i],
                zero_division=self.zero_division,
            )
            f1_values[i] = f1score(
                precision_value=precision_values[i],
                recall_value=recall_values[i],
            )

        # per-class metrics
        for metric_name, metric_value in zip(
            ("precision", "recall", "f1", "support"),
            (
                precision_values,
                recall_values,
                f1_values,
                self.statistics["support"],
            ),
        ):
            for i in range(self.num_classes):
                self.metrics[f"{metric_name}/class_{i+1:02d}"] = metric_value[
                    i
                ]

        # macro metrics
        macro_average = self._compute_weighted_average(
            precision_values=precision_values,
            recall_values=recall_values,
            f1_values=f1_values,
            weights=1 / self.num_classes,
        )
        self.metrics.update(
            {f"{name}/macro": value for name, value in macro_average.items()}
        )

        # weighted metrics
        weights = self.statistics["support"] / self.statistics["support"].sum()
        weighted_average = self._compute_weighted_average(
            precision_values=precision_values,
            recall_values=recall_values,
            f1_values=f1_values,
            weights=weights,
        )
        self.metrics.update(
            {
                f"{name}/weighted": value
                for name, value in weighted_average.items()
            }
        )

        # micro metrics
        micro_average = self._compute_micro_average()
        self.metrics.update(
            {f"{name}/micro": value for name, value in micro_average.items()}
        )

        return self.metrics


class MultiClassPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multiclass classification.
    Counts metrics with macro, micro and weighted average.
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        num_classes: int = 2,
        zero_division: int = 0,
    ):
        """
        Init MultiClassPrecisionRecallF1SupportMetric instance

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
            num_classes: number of classes in loader's dataset
            zero_division: value to set in case of zero division during metrics
                (precision, recall) computation; should be one of 0 or 1
        """
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            zero_division=zero_division,
            mode="multiclass",
        )


class MultiLabelPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multilabel classification.
    Counts metrics with macro, micro and weighted average.
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        num_classes: int = 2,
        zero_division: int = 0,
    ):
        """
        Init MultiLabelPrecisionRecallF1SupportMetric instance

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
            num_classes: number of classes in loader's dataset
            zero_division: value to set in case of zero division during metrics
                (precision, recall) computation; should be one of 0 or 1
        """
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            zero_division=zero_division,
            mode="multilabel",
        )


class BinaryPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """Precision, recall, f1_score and support metrics for binary classification."""

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        zero_division: int = 0,
    ):
        """
        Init BinaryPrecisionRecallF1SupportMetric instance

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
            zero_division: value to set in case of zero division during metrics
                (precision, recall) computation; should be one of 0 or 1
        """
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=2,
            zero_division=zero_division,
            mode="binary",
        )

    def reset(self, num_batches: int, num_samples: int) -> None:
        """Reset all the statistics and metrics fields."""
        self.statistics = defaultdict(float)
        self.metrics = defaultdict(float)

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics with accumulated statistics

        Returns:
            dict of metrics
        """
        self.metrics["precision"] = precision(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            zero_division=self.zero_division,
        )
        self.metrics["recall"] = recall(
            tp=self.statistics["tp"],
            fn=self.statistics["fn"],
            zero_division=self.zero_division,
        )
        self.metrics["f1"] = f1score(
            precision_value=self.metrics["precision"],
            recall_value=self.metrics["recall"],
        )
        return self.metrics


__all__ = [
    "precision_recall_fbeta_support",
    "MultiClassPrecisionRecallF1SupportMetric",
    "MultiLabelPrecisionRecallF1SupportMetric",
    "BinaryPrecisionRecallF1SupportMetric",
]
