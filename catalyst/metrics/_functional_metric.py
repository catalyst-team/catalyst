from typing import Callable, Dict, Iterable

import torch

from catalyst.metrics import AccumulativeMetric, ICallbackBatchMetric, ICallbackLoaderMetric
from catalyst.metrics._additive import AdditiveMetric


class FunctionalBatchMetric(ICallbackBatchMetric):
    """Class for custom **batch-based** metrics in a functional way.

    Args:
        metric_fn: metric function, that get outputs, targets and return score as torch.Tensor
        metric_key: metric name
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix

    .. note::

        Loader metrics calculated as average over all batch metrics.

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics
        import sklearn.metrics

        outputs = torch.tensor([1, 0, 2, 1])
        targets = torch.tensor([3, 0, 2, 2])

        metric = metrics.FunctionalBatchMetric(
            metric_fn=sklearn.metrics.accuracy_score,
            metric_key="sk_accuracy",
        )
        metric.reset()

        metric.update(batch_size=len(outputs), y_pred=outputs, y_true=targets)
        metric.compute()
        # (0.5, 0.0)  # mean, std

        metric.compute_key_value()
        # {'sk_accuracy': 0.5, 'sk_accuracy/mean': 0.5, 'sk_accuracy/std': 0.0}

    """

    def __init__(
        self,
        metric_fn: Callable,
        metric_key: str,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_fn = metric_fn
        self.metric_name = f"{self.prefix}{metric_key}{self.suffix}"
        self.additive_metric = AdditiveMetric()

    def reset(self):
        """Reset all statistics"""
        self.additive_metric.reset()

    def update(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        """
        Calculate metric and update average metric

        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn

        Returns:
            custom metric
        """
        value = self.metric_fn(*args, **kwargs)
        self.additive_metric.update(float(value), batch_size)
        return value

    def update_key_value(self, batch_size: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Calculate metric and update average metric

        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn

        Returns:
            Dict with one element-custom metric
        """
        value = self.update(batch_size, *args, **kwargs)
        return {f"{self.metric_name}": value}

    def compute(self) -> torch.Tensor:
        """
        Get metric average over all examples

        Returns:
            custom metric
        """
        return self.additive_metric.compute()

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Get metric average over all examples

        Returns:
            Dict with one element-custom metric
        """
        mean, std = self.compute()
        return {
            self.metric_name: mean,
            f"{self.metric_name}/mean": mean,
            f"{self.metric_name}/std": std,
        }


class FunctionalLoaderMetric(ICallbackLoaderMetric):
    """Class for custom **loader-based** metrics in a functional way.

    Args:
        metric_fn: metric function, that get outputs, targets and return score as torch.Tensor
        metric_key: metric name
        accumulative_fields: list of keys to accumulate data from batch
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix

    .. note::

        Metrics are calculated over all samples.

    Examples:

    .. code-block:: python

        from functools import partial
        import torch
        from catalyst import metrics
        import sklearn.metrics

        targets = torch.tensor([3, 0, 2, 2, 1])
        outputs = torch.rand((len(targets), targets.max()+1)).softmax(1)

        metric = metrics.FunctionalLoaderMetric(
            metric_fn=partial(
                sklearn.metrics.roc_auc_score, average="macro", multi_class="ovr"
            ),
            metric_key="sk_auc",
            accumulative_fields=['y_score','y_true'],

        )
        metric.reset(len(outputs), len(outputs))

        metric.update(y_score=outputs, y_true=targets)
        metric.compute()
        # ...

        metric.compute_key_value()
        # {'sk_auc': ...}

    """

    def __init__(
        self,
        metric_fn: Callable,
        metric_key: str,
        accumulative_fields: Iterable[str] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_fn = metric_fn
        self.metric_name = f"{self.prefix}{metric_key}{self.suffix}"
        self.accumulative_metric = AccumulativeMetric(
            keys=accumulative_fields, compute_on_call=compute_on_call
        )

    def reset(self, num_batches: int, num_samples: int) -> None:
        """
        Reset metrics fields

        Args:
            num_batches: expected number of batches
            num_samples: expected number of samples to accumulate
        """
        self.accumulative_metric.reset(num_batches, num_samples)

    def update(self, **kwargs) -> None:
        """
        Update storage

        Args:
            **kwargs: ``self.metric_fn`` inputs to store
        """
        self.accumulative_metric.update(**kwargs)

    def compute(self) -> torch.Tensor:
        """
        Get metric for the whole loader

        Returns:
            custom metric
        """
        stored_values = self.accumulative_metric.compute()
        return self.metric_fn(**stored_values)

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Get metric for the whole loader

        Returns:
            Dict with one element-custom metric
        """
        return {self.metric_name: self.compute()}


__all__ = ["FunctionalBatchMetric", "FunctionalLoaderMetric"]
