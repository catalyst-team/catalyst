# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Any, Callable, Dict, List, Union
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

import numpy as np

import torch

from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools import meters

logger = logging.getLogger(__name__)


class IMetricCallback(ABC, Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        prefix: str,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        multiplier: float = 1.0,
        **metrics_kwargs,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.multiplier = multiplier
        self.metrics_kwargs = metrics_kwargs

        self._get_input = utils.get_dictkey_auto_fn(self.input_key)
        self._get_output = utils.get_dictkey_auto_fn(self.output_key)

        kv_types = (dict, tuple, list, type(None))

        is_value_input = (
            isinstance(self.input_key, str) and self.input_key != "__all__"
        )
        is_value_output = (
            isinstance(self.output_key, str) and self.output_key != "__all__"
        )
        is_kv_input = (
            isinstance(self.input_key, kv_types) or self.input_key == "__all__"
        )
        is_kv_output = (
            isinstance(self.output_key, kv_types)
            or self.output_key == "__all__"
        )

        # @TODO: fix to only KV usage
        if hasattr(self, "_compute_metric"):
            pass  # overridden in descendants
        elif is_value_input and is_value_output:
            self._compute_metric = self._compute_metric_value
        elif is_kv_input and is_kv_output:
            self._compute_metric = self._compute_metric_key_value
        else:
            raise NotImplementedError()

    @property
    @abstractmethod
    def metric_fn(self):
        """@TODO: Docs. Contribution is welcome."""
        pass

    def _compute_metric_value(self, output: Dict, input: Dict):
        """@TODO: Docs. Contribution is welcome."""
        output = self._get_output(output, self.output_key)
        input = self._get_input(input, self.input_key)

        metric = self.metric_fn(output, input, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, output: Dict, input: Dict):
        """@TODO: Docs. Contribution is welcome."""
        output = self._get_output(output, self.output_key)
        input = self._get_input(input, self.input_key)

        metric = self.metric_fn(**output, **input, **self.metrics_kwargs)
        return metric

    def _process_computed_metric(self, metric) -> Dict:
        """@TODO: Docs. Contribution is welcome."""
        if isinstance(metric, dict):
            metric = {
                f"{self.prefix}{key}": value * self.multiplier
                for key, value in metric.items()
            }
        elif isinstance(metric, (float, int, torch.Tensor)):
            metric = {f"{self.prefix}": metric * self.multiplier}
        else:
            raise NotImplementedError()
        return metric


class IBatchMetricCallback(IMetricCallback):
    """@TODO: Docs. Contribution is welcome."""

    def on_batch_end(self, runner: IRunner) -> None:
        """Computes metrics and add them to batch metrics."""
        metrics = self._compute_metric(runner.output, runner.input)
        metrics = self._process_computed_metric(metrics)
        runner.batch_metrics.update(**metrics)


class ILoaderMetricCallback(IMetricCallback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, **kwargs):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(**kwargs)

        self.input = defaultdict(lambda: [])
        self.output = defaultdict(lambda: [])

    def on_loader_start(self, runner: IRunner):
        """Reinitialises internal storages."""
        self.input = defaultdict(lambda: [])
        self.output = defaultdict(lambda: [])

    def on_batch_end(self, runner: IRunner) -> None:
        """Stores new input/output for the metric computation."""
        output = self._get_output(runner.output, self.output_key)
        input = self._get_input(runner.input, self.input_key)

        for data, storage in zip((input, output), (self.input, self.output)):
            if isinstance(data, dict):
                for key, value in data.items():
                    storage[key].append(value.detach().cpu().numpy())
            else:
                storage["_data"].append(data.detach().cpu().numpy())

    def on_loader_end(self, runner: IRunner):
        """@TODO: Docs. Contribution is welcome."""
        input = {
            key: torch.from_numpy(np.concatenate(self.input[key], axis=0))
            for key in self.input
        }
        output = {
            key: torch.from_numpy(np.concatenate(self.output[key], axis=0))
            for key in self.output
        }

        input = {self.input_key: input["_data"]} if len(input) == 1 else input
        output = (
            {self.output_key: output["_data"]} if len(output) == 1 else output
        )

        metrics = self._compute_metric(output, input)
        metrics = self._process_computed_metric(metrics)
        runner.loader_metrics.update(**metrics)


class BatchMetricCallback(IBatchMetricCallback):
    """A callback that returns single metric on `runner.on_batch_end`."""

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metric_kwargs,
        )
        self.metric = metric_fn

    @property
    def metric_fn(self):
        """@TODO: Docs. Contribution is welcome."""
        return self.metric


class LoaderMetricCallback(ILoaderMetricCallback):
    """A callback that returns single metric on `runner.on_batch_end`."""

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metric_kwargs,
        )
        self.metric = metric_fn

    @property
    def metric_fn(self):
        """@TODO: Docs. Contribution is welcome."""
        return self.metric


class MetricAggregationCallback(Callback):
    """A callback to aggregate several metrics in one value."""

    def __init__(
        self,
        prefix: str,
        metrics: Union[str, List[str], Dict[str, float]] = None,
        mode: str = "mean",
        scope: str = "batch",
        multiplier: float = 1.0,
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated metric.
            metrics (Union[str, List[str], Dict[str, float]]): If not None,
                it aggregates only the values from the metric by these keys.
                for ``weighted_sum`` aggregation it must be a Dict[str, float].
            mode (str): function for aggregation.
                Must be either ``sum``, ``mean`` or ``weighted_sum``.
            multiplier (float): scale factor for the aggregated metric.
        """
        super().__init__(
            order=CallbackOrder.metric_aggregation, node=CallbackNode.all
        )

        if prefix is None or not isinstance(prefix, str):
            raise ValueError("prefix must be str")

        if mode in ("sum", "mean"):
            if metrics is not None and not isinstance(metrics, list):
                raise ValueError(
                    "For `sum` or `mean` mode the metrics must be "
                    "None or list or str (not dict)"
                )
        elif mode in ("weighted_sum", "weighted_mean"):
            if metrics is None or not isinstance(metrics, dict):
                raise ValueError(
                    "For `weighted_sum` or `weighted_mean` mode "
                    "the metrics must be specified "
                    "and must be a dict"
                )
        else:
            raise NotImplementedError(
                "mode must be `sum`, `mean` "
                "or `weighted_sum` or `weighted_mean`"
            )

        assert scope in ("batch", "loader", "epoch")

        if isinstance(metrics, str):
            metrics = [metrics]

        self.prefix = prefix
        self.metrics = metrics
        self.mode = mode
        self.scope = scope
        self.multiplier = multiplier

        if mode in ("sum", "weighted_sum", "weighted_mean"):
            self.aggregation_fn = (
                lambda x: torch.sum(torch.stack(x)) * multiplier
            )
            if mode == "weighted_mean":
                weights_sum = sum(metrics.items())
                self.metrics = {
                    key: weight / weights_sum
                    for key, weight in metrics.items()
                }
        elif mode == "mean":
            self.aggregation_fn = (
                lambda x: torch.mean(torch.stack(x)) * multiplier
            )

    def _preprocess(self, metrics: Any) -> List[float]:
        if self.metrics is not None:
            if self.mode == "weighted_sum":
                result = [
                    metrics[key] * value for key, value in self.metrics.items()
                ]
            else:
                result = [metrics[key] for key in self.metrics]
        else:
            result = list(metrics.values())
        return result

    def _process_metrics(self, metrics: Dict):
        metrics_ = self._preprocess(metrics)
        metric_ = self.aggregation_fn(metrics_)
        metrics[self.prefix] = metric_

    def on_batch_end(self, runner: IRunner) -> None:
        """Computes the metric and add it to the metrics.

        Args:
            runner (IRunner): current runner
        """
        if self.scope == "batch":
            self._process_metrics(runner.batch_metrics)

    def on_loader_end(self, runner: IRunner):
        if self.scope == "loader":
            self._process_metrics(runner.loader_metrics)

    def on_epoch_end(self, runner: IRunner):
        if self.scope == "epoch":
            self._process_metrics(runner.epoch_metrics)


class MetricManagerCallback(Callback):
    """
    Prepares metrics for logging, transferring values from PyTorch to numpy.
    """

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            order=CallbackOrder.logging - 1, node=CallbackNode.all,
        )
        self.meters: Dict[str, meters.AverageValueMeter] = None

    @staticmethod
    def to_single_value(value: Any) -> float:
        """@TODO: Docs. Contribution is welcome."""
        if hasattr(value, "item"):
            value = value.item()

        value = float(value)
        return value

    @staticmethod
    def _process_metrics(metrics: Dict[str, Any]):
        output = {}
        for key, value in metrics.items():
            value = utils.get_distributed_mean(value)
            value = MetricManagerCallback.to_single_value(value)
            output[key] = value
        return output

    def on_epoch_start(self, runner: IRunner) -> None:
        """Epoch start hook.
        Args:
            runner (IRunner): current runner
        """
        runner.epoch_metrics = defaultdict(None)

    def on_loader_start(self, runner: IRunner) -> None:
        """Loader start hook.
        Args:
            runner (IRunner): current runner
        """
        runner.loader_metrics = defaultdict(None)
        self.meters = defaultdict(meters.AverageValueMeter)

    def on_batch_start(self, runner: IRunner) -> None:
        """Batch start hook.
        Args:
            runner (IRunner): current runner
        """
        runner.batch_metrics = defaultdict(None)

    def on_batch_end(self, runner: IRunner) -> None:
        """Batch end hook.
        Args:
            runner (IRunner): current runner
        """
        runner.batch_metrics = self._process_metrics(runner.batch_metrics)
        for key, value in runner.batch_metrics.items():
            self.meters[key].add(value, runner.batch_size)

    def on_loader_end(self, runner: IRunner) -> None:
        """Loader end hook.
        Args:
            runner (IRunner): current runner
        """
        for key, value in self.meters.items():
            value = value.mean
            runner.loader_metrics[key] = value
        for key, value in runner.loader_metrics.items():
            runner.epoch_metrics[f"{runner.loader_name}_{key}"] = value


# backward compatibility
MetricCallback = BatchMetricCallback

__all__ = [
    "IMetricCallback",
    "IBatchMetricCallback",
    "ILoaderMetricCallback",
    "BatchMetricCallback",
    "LoaderMetricCallback",
    "MetricCallback",
    "MetricAggregationCallback",
    "MetricManagerCallback",
]
