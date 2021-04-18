from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._classification import get_aggregated_metrics, get_binary_metrics
from catalyst.metrics.functional._misc import (
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)
from catalyst.utils.distributed import all_gather, get_rank


class StatisticsMetric(ICallbackBatchMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from data.

    It can work in binary, multiclass and multilabel classification tasks.

    Args:
        mode: one of "binary", "multiclass" and "multilabel"
        num_classes: number of classes
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        mode: str,
        num_classes: int = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init params

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
            raise ValueError("Mode should be one of 'binary', 'multiclass', 'multilabel'")

        self.num_classes = num_classes
        self.statistics = None
        self._is_ddp = False
        self.reset()

    # multiprocessing could not handle lamdas, so..
    def _mp_hack(self):
        return np.zeros(shape=(self.num_classes,))

    def reset(self) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(self._mp_hack)
        self._is_ddp = get_rank() > -1

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Tuple[int, int, int, int, int], Tuple[Any, Any, Any, Any, Any]]:
        """
        Compute statistics from outputs and targets, update accumulated statistics with new values.

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            Tuple of int or array: true negative, false positive, false
                negative, true positive and support statistics
        """
        tn, fp, fn, tp, support = self.statistics_fn(
            outputs=outputs.cpu().detach(), targets=targets.cpu().detach(),
        )

        tn = tn.numpy()
        fp = fp.numpy()
        fn = fn.numpy()
        tp = tp.numpy()
        support = support.numpy()

        self.statistics["tn"] += tn
        self.statistics["fp"] += fp
        self.statistics["fn"] += fn
        self.statistics["tp"] += tp
        self.statistics["support"] += support

        return tn, fp, fn, tp, support

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics and return statistics intermediate result

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of statistics for current input
        """
        tn, fp, fn, tp, support = self.update(outputs=outputs, targets=targets)
        return {"fn": fn, "fp": fp, "support": support, "tn": tn, "tp": tp}

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

    Args:
        mode: one of "binary", "multiclass" and "multilabel"
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: TODO
        suffix: TODO
    """

    def __init__(
        self,
        mode: str,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ) -> None:
        """Init PrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            mode=mode,
        )
        self.zero_division = zero_division
        self.reset()

    def _convert_metrics_to_kv(self, per_class, micro, macro, weighted) -> Dict[str, float]:
        """
        Convert metrics aggregation to key-value format

        Args:
            per_class: per-class metrics, array of shape (4, self.num_classes)
                of precision, recall, f1 and support metrics
            micro: micro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            macro: macro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            weighted: weighted averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics

        Returns:
            dict of key-value metrics
        """
        kv_metrics = {}
        for aggregation_name, aggregated_metrics in zip(
            ("_micro", "_macro", "_weighted"), (micro, macro, weighted)
        ):
            metrics = {
                f"{metric_name}/{aggregation_name}": metric_value
                for metric_name, metric_value in zip(
                    ("precision", "recall", "f1"), aggregated_metrics[:-1]
                )
            }
            kv_metrics.update(metrics)

        per_class_metrics = {
            f"{metric_name}/class_{i:02d}": metric_value[i]
            for metric_name, metric_value in zip(
                ("precision", "recall", "f1", "support"), per_class
            )
            for i in range(self.num_classes)  # noqa: WPS361
        }
        kv_metrics.update(per_class_metrics)
        return kv_metrics

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            tuple of metrics intermediate results with per-class, micro, macro and
                weighted averaging
        """
        tn, fp, fn, tp, support = super().update(outputs=outputs, targets=targets)
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=tp, fp=fp, fn=fn, support=support, zero_division=self.zero_division,
        )
        return per_class, micro, macro, weighted

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of metrics intermediate results
        """
        per_class, micro, macro, weighted = self.update(outputs=outputs, targets=targets)
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            list of aggregated metrics: per-class, micro, macro and weighted averaging of
                precision, recall, f1 score and support metrics
        """
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            support=self.statistics["support"],
            zero_division=self.zero_division,
        )
        return per_class, micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics
        """
        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            for key in self.statistics:
                value: List[np.ndarray] = all_gather(self.statistics[key])
                value: np.ndarray = np.sum(np.vstack(value), axis=0)
                self.statistics[key] = value

        per_class, micro, macro, weighted = self.compute()
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics


class BinaryPrecisionRecallF1Metric(StatisticsMetric):
    """Precision, recall, f1_score and support metrics for binary classification.

    Args:
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init BinaryPrecisionRecallF1SupportMetric instance"""
        super().__init__(
            num_classes=2,
            mode="binary",
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
        )
        self.zero_division = zero_division
        self.reset()

    @staticmethod
    def _convert_metrics_to_kv(
        precision_value: float, recall_value: float, f1_value: float
    ) -> Dict[str, float]:
        """
        Convert list of metrics to key-value

        Args:
            precision_value: precision value
            recall_value: recall value
            f1_value: f1 value

        Returns:
            dict of metrics
        """
        kv_metrics = {
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
        }
        return kv_metrics

    def reset(self) -> None:
        """Reset all the statistics and metrics fields."""
        self.statistics = defaultdict(float)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
        """
        Update statistics and return metrics intermediate results

        Args:
            outputs: predicted labels
            targets: target labels

        Returns:
            tuple of intermediate metrics: precision, recall, f1 score
        """
        tn, fp, fn, tp, support = super().update(outputs=outputs, targets=targets)
        precision_value, recall_value, f1_value = get_binary_metrics(
            tp=tp, fp=fp, fn=fn, zero_division=self.zero_division
        )
        return precision_value, recall_value, f1_value

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics and return metrics intermediate results

        Args:
            outputs: predicted labels
            targets: target labels

        Returns:
            dict of intermediate metrics
        """
        precision_value, recall_value, f1_value = self.update(outputs=outputs, targets=targets)
        kv_metrics = self._convert_metrics_to_kv(
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value,
        )
        return kv_metrics

    def compute(self) -> Tuple[float, float, float]:
        """
        Compute metrics with accumulated statistics

        Returns:
            tuple of metrics: precision, recall, f1 score
        """
        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            for key in self.statistics:
                value: List[float] = all_gather(self.statistics[key])
                value: float = sum(value)
                self.statistics[key] = value

        precision_value, recall_value, f1_value = get_binary_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            zero_division=self.zero_division,
        )
        return precision_value, recall_value, f1_value

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute metrics with all accumulated statistics

        Returns:
            dict of metrics
        """
        precision_value, recall_value, f1_value = self.compute()
        kv_metrics = self._convert_metrics_to_kv(
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value,
        )
        return kv_metrics


class MulticlassPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multiclass classification.
    Counts metrics with macro, micro and weighted average.

    Args:
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init MultiClassPrecisionRecallF1SupportMetric instance"""
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

    Args:
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init MultiLabelPrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            zero_division=zero_division,
            mode="multilabel",
        )


__all__ = [
    "BinaryPrecisionRecallF1Metric",
    "MulticlassPrecisionRecallF1SupportMetric",
    "MultilabelPrecisionRecallF1SupportMetric",
]
