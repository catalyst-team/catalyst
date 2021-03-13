from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from catalyst.metrics._additive import AdditiveValueMetric
from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._accuracy import accuracy, multilabel_accuracy
from catalyst.metrics.functional._misc import get_default_topk_args


class AccuracyMetric(ICallbackBatchMetric):
    """
    This metric computes accuracy for multiclass classification case.
    It computes mean value of accuracy and it's approximate std value
    (note that it's not a real accuracy std but std of accuracy over batch mean values).

    Args:
        topk_args: list of `topk` for accuracy@topk computing
        num_classes: number of classes
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        topk_args: List[int] = None,
        num_classes: int = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init AccuracyMetric"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name_mean = f"{self.prefix}accuracy{self.suffix}"
        self.metric_name_std = f"{self.prefix}accuracy{self.suffix}/std"
        self.topk_args: List[int] = topk_args or get_default_topk_args(num_classes)
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        """Reset all fields"""
        for metric in self.additive_metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Updates metric value with accuracy for new data and return intermediate metrics values.

        Args:
            logits: tensor of logits
            targets: tensor of targets

        Returns:
            list of accuracy@k values
        """
        values = accuracy(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update metric value with accuracy for new data and return intermediate metrics
        values in key-value format.

        Args:
            logits: tensor of logits
            targets: tensor of targets

        Returns:
            dict of accuracy@k values
        """
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[f"{self.prefix}accuracy01{self.suffix}"]
        return output

    def compute(self) -> Tuple[List[float], List[float]]:
        """
        Compute accuracy for all data

        Returns:
            list of mean values, list of std values
        """
        means, stds = zip(*(metric.compute() for metric in self.additive_metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute accuracy for all data and return results in key-value format

        Returns:
            dict of metrics
        """
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_std = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk_args, stds)
        }
        output_mean[self.metric_name_mean] = output_mean[f"{self.prefix}accuracy01{self.suffix}"]
        output_std[self.metric_name_std] = output_std[f"{self.prefix}accuracy01{self.suffix}/std"]
        return {**output_mean, **output_std}


class MultilabelAccuracyMetric(AdditiveValueMetric, ICallbackBatchMetric):
    """
    This metric computes accuracy for multilabel classification case.
    It computes mean value of accuracy and it's approximate std value
    (note that it's not a real accuracy std but std of accuracy over batch mean values).

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
        threshold: thresholds for model scores
    """

    def __init__(
        self,
        threshold: Union[float, torch.Tensor] = 0.5,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init MultilabelAccuracyMetric"""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.metric_name_mean = f"{self.prefix}accuracy{self.suffix}"
        self.metric_name_std = f"{self.prefix}accuracy{self.suffix}/std"
        self.threshold = threshold

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Update metric value with accuracy for new data and return intermediate metric value.

        Args:
            outputs: tensor of outputs
            targets: tensor of true answers

        Returns:
            accuracy metric for outputs and targets
        """
        metric = multilabel_accuracy(
            outputs=outputs, targets=targets, threshold=self.threshold
        ).item()
        super().update(value=metric, num_samples=np.prod(targets.shape))
        return metric

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update metric value with accuracy for new data and return intermediate metric
        value in key-value format.

        Args:
            outputs: tensor of outputs
            targets: tensor of true answers

        Returns:
            accuracy metric for outputs and targets
        """
        metric = self.update(outputs=outputs, targets=targets)
        return {self.metric_name_mean: metric}

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute accuracy for all data and return results in key-value format

        Returns:
            dict of metrics
        """
        metric_mean, metric_std = self.compute()
        return {
            self.metric_name_mean: metric_mean,
            self.metric_name_std: metric_std,
        }


__all__ = ["AccuracyMetric", "MultilabelAccuracyMetric"]
