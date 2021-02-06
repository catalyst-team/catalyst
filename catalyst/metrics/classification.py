from typing import Any, Dict, Optional, Tuple, Union
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from catalyst.metrics.functional.classification import (
    f1score,
    precision,
    precision_recall_fbeta_support,
    recall,
)
from catalyst.metrics.functional.misc import (
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)
from catalyst.metrics.metric import ICallbackBatchMetric, ICallbackLoaderMetric


# @TODO: make ICallbackBatchMetric
class StatisticsMetric(ICallbackLoaderMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from data.

    It can work in binary, multiclass and multilabel classification tasks.
    """

    def __init__(
        self,
        num_classes: int,
        mode: str,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
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
            self.statistics_fn = partial(get_multiclass_statistics, num_classes=num_classes)
        elif mode == "multilabel":
            self.statistics_fn = get_multilabel_statistics
        else:
            raise ValueError(f'Mode should be one of "binary", "multiclass", "multilabel".')

        self.num_classes = num_classes
        self.statistics = None
        self.reset(num_batches=0, num_samples=0)

    def reset(self, num_batches: int, num_samples: int) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(lambda: np.zeros(shape=(self.num_classes,)))

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
        num_classes: int,
        mode: str,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ) -> None:
        """
        Init PrecisionRecallF1SupportMetric instance

        Args:
            num_classes: number of classes in loader's dataset
            mode: one of "binary", "multiclass" and "multilabel"
            zero_division: value to set in case of zero division during metrics
                (precision, recall) computation; should be one of 0 or 1
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
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
            ("precision", "recall", "f1"), (precision_values, recall_values, f1_values),
        ):
            weighted_metrics[metric_name] = (metric_value * weights).sum()
        return weighted_metrics

    def _compute_micro_average(
        self, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray, tp: np.ndarray, zero_division: int
    ) -> Dict[str, float]:
        """
        Compute micro precision, recall and f1_score values with statistics.
        Returns:
            dict of averaged metrics
        """
        micro_metrics = defaultdict(float)
        micro_metrics["precision"] = precision(
            tp=tp.sum(), fp=fp.sum(), zero_division=zero_division
        )
        micro_metrics["recall"] = recall(tp=tp.sum(), fn=fn.sum(), zero_division=zero_division)
        micro_metrics["f1"] = f1score(
            precision_value=micro_metrics["precision"], recall_value=micro_metrics["recall"],
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
                precision_value=precision_values[i], recall_value=recall_values[i],
            )

        # per-class metrics
        for metric_name, metric_value in zip(
            ("precision", "recall", "f1", "support"),
            (precision_values, recall_values, f1_values, self.statistics["support"]),
        ):
            for i in range(self.num_classes):
                self.metrics[f"{metric_name}/class_{i:02d}"] = metric_value[i]

        # macro metrics
        macro_average = self._compute_weighted_average(
            precision_values=precision_values,
            recall_values=recall_values,
            f1_values=f1_values,
            weights=1.0 / self.num_classes,
        )
        self.metrics.update({f"{name}/macro": value for name, value in macro_average.items()})

        # weighted metrics
        weights = self.statistics["support"] / self.statistics["support"].sum()
        weighted_average = self._compute_weighted_average(
            precision_values=precision_values,
            recall_values=recall_values,
            f1_values=f1_values,
            weights=weights,
        )
        self.metrics.update(
            {f"{name}/weighted": value for name, value in weighted_average.items()}
        )

        # micro metrics
        micro_average = self._compute_micro_average(
            tn=self.statistics["tn"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            tp=self.statistics["tp"],
            zero_division=self.zero_division,
        )
        self.metrics.update({f"{name}/micro": value for name, value in micro_average.items()})

        return self.metrics


class BinaryPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """Precision, recall, f1_score and support metrics for binary classification."""

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
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
            num_classes=2,
            mode="binary",
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            zero_division=zero_division,
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
            tp=self.statistics["tp"], fp=self.statistics["fp"], zero_division=self.zero_division,
        )
        self.metrics["recall"] = recall(
            tp=self.statistics["tp"], fn=self.statistics["fn"], zero_division=self.zero_division,
        )
        self.metrics["f1"] = f1score(
            precision_value=self.metrics["precision"], recall_value=self.metrics["recall"],
        )
        return self.metrics


class MulticlassPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multiclass classification.
    Counts metrics with macro, micro and weighted average.
    """

    def __init__(
        self,
        num_classes: int,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
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


class MultilabelPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multilabel classification.
    Counts metrics with macro, micro and weighted average.
    """

    def __init__(
        self,
        num_classes: int,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
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


__all__ = [
    "PrecisionRecallF1SupportMetric",
    "BinaryPrecisionRecallF1SupportMetric",
    "MulticlassPrecisionRecallF1SupportMetric",
    "MultilabelPrecisionRecallF1SupportMetric",
]
