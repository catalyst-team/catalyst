from typing import Any, Callable, Dict, List, Union
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

import torch

from catalyst.core import Callback, CallbackNode, CallbackOrder, State, utils
from catalyst.utils import meters

logger = logging.getLogger(__name__)


class _MetricCallback(ABC, Callback):
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
        super().__init__(order=CallbackOrder.Metric, node=CallbackNode.All)
        self.prefix = prefix
        # self.metric_fn = partial(metric_fn, **metric_params)
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

    def _compute_metric_value(self, state: State):
        output = self._get_output(state.batch_out, self.output_key)
        input = self._get_input(state.batch_in, self.input_key)

        metric = self.metric_fn(output, input, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, state: State):
        output = self._get_output(state.batch_out, self.output_key)
        input = self._get_input(state.batch_in, self.input_key)

        metric = self.metric_fn(**output, **input, **self.metrics_kwargs)
        return metric

    def on_batch_end(self, state: State) -> None:
        """Computes the metric and add it to batch metrics."""
        metric = self._compute_metric(state) * self.multiplier
        state.batch_metrics[self.prefix] = metric


class MetricCallback(_MetricCallback):
    """A callback that returns single metric on `state.on_batch_end`."""

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


class MultiMetricCallback(MetricCallback):
    """A callback that returns multiple metrics on `state.on_batch_end`."""

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        list_args: List,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        multiplier: float = 1.0,
        **metrics_kwargs,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            prefix=prefix,
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metrics_kwargs,
        )
        self.list_args = list_args

    def on_batch_end(self, state: State) -> None:
        """Batch end hook.

        Args:
            state (State): current state
        """
        metrics_ = self._compute_metric(state)

        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            state.batch_metrics[key] = metric * self.multiplier


class MetricAggregationCallback(Callback):
    """A callback to aggregate several metrics in one value."""

    def __init__(
        self,
        prefix: str,
        metrics: Union[str, List[str], Dict[str, float]] = None,
        mode: str = "mean",
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
            order=CallbackOrder.MetricAggregation, node=CallbackNode.All
        )

        if prefix is None or not isinstance(prefix, str):
            raise ValueError("prefix must be str")

        if mode in ("sum", "mean"):
            if metrics is not None and not isinstance(metrics, list):
                raise ValueError(
                    "For `sum` or `mean` mode the loss_keys must be "
                    "None or list or str (not dict)"
                )
        elif mode in ("weighted_sum", "weighted_mean"):
            if metrics is None or not isinstance(metrics, dict):
                raise ValueError(
                    "For `weighted_sum` or `weighted_mean` mode "
                    "the loss_keys must be specified "
                    "and must be a dict"
                )
        else:
            raise NotImplementedError(
                "mode must be `sum`, `mean` "
                "or `weighted_sum` or `weighted_mean`"
            )

        if isinstance(metrics, str):
            metrics = [metrics]

        self.prefix = prefix
        self.metrics = metrics
        self.mode = mode
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

    def on_batch_end(self, state: State) -> None:
        """Computes the metric and add it to the metrics.

        Args:
            state (State): current state
        """
        metrics = self._preprocess(state.batch_metrics)
        metric = self.aggregation_fn(metrics)
        state.batch_metrics[self.prefix] = metric


class MetricManagerCallback(Callback):
    """
    Prepares metrics for logging, transferring values from PyTorch to numpy.
    """

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            order=CallbackOrder.Logging - 1, node=CallbackNode.All,
        )
        self.meters: Dict[str, meters.AverageValueMeter] = None

    @staticmethod
    def _to_single_value(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()

        value = float(value)
        return value

    @staticmethod
    def _process_metrics(metrics: Dict[str, Any]):
        output = {}
        for key, value in metrics.items():
            value = utils.get_distributed_mean(value)
            value = MetricManagerCallback._to_single_value(value)
            output[key] = value
        return output

    def on_epoch_start(self, state: State) -> None:
        """Epoch start hook.

        Args:
            state (State): current state
        """
        state.epoch_metrics = defaultdict(None)

    def on_loader_start(self, state: State) -> None:
        """Loader start hook.

        Args:
            state (State): current state
        """
        state.loader_metrics = defaultdict(None)
        self.meters = defaultdict(meters.AverageValueMeter)

    def on_loader_end(self, state: State) -> None:
        """Loader end hook.

        Args:
            state (State): current state
        """
        for key, value in self.meters.items():
            value = value.mean
            state.loader_metrics[key] = value
        for key, value in state.loader_metrics.items():
            state.epoch_metrics[f"{state.loader_name}_{key}"] = value

    def on_batch_start(self, state: State) -> None:
        """Batch start hook.

        Args:
            state (State): current state
        """
        state.batch_metrics = defaultdict(None)

    def on_batch_end(self, state: State) -> None:
        """Batch end hook.

        Args:
            state (State): current state
        """
        state.batch_metrics = self._process_metrics(state.batch_metrics)
        for key, value in state.batch_metrics.items():
            self.meters[key].add(value)


__all__ = [
    "_MetricCallback",
    "MetricCallback",
    "MultiMetricCallback",
    "MetricAggregationCallback",
    "MetricManagerCallback",
]
