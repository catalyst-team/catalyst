from typing import (  # isort:skip
    Any, Callable, List, Dict, Union, TYPE_CHECKING  # isort:skip
)  # isort:skip
from functools import partial
import logging

import torch

from catalyst.core import Callback, CallbackOrder

if TYPE_CHECKING:
    from catalyst.core import _State

logger = logging.getLogger(__name__)


class MetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = partial(metric_fn, **metric_params)
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: "_State"):
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]
        metric = self.metric_fn(outputs, targets)
        state.batch_metrics[self.prefix] = metric


class MultiMetricCallback(Callback):
    """
    A callback that returns multiple metrics on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        list_args: List,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = partial(metric_fn, **metric_params)
        self.list_args = list_args
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: "_State"):
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]

        metrics_ = self.metric_fn(outputs, targets, self.list_args)

        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            state.batch_metrics[key] = metric


class MetricAggregatorCallback(Callback):
    """
    A callback to aggregate several metrics in one value.
    """
    def __init__(
        self,
        prefix: str,
        metrics: Union[str, List[str], Dict[str, float]] = None,
        mode: str = "sum",
        multiplier: float = 1.0
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
        super().__init__(CallbackOrder.Metric + 1)

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
            self.aggregation_fn = \
                lambda x: torch.sum(torch.stack(x)) * multiplier
            if mode == "weighted_mean":
                weights_sum = sum(metrics.items())
                self.metrics = {
                    key: weight / weights_sum
                    for key, weight in metrics.items()
                }
        elif mode == "mean":
            self.aggregation_fn = \
                lambda x: torch.mean(torch.stack(x)) * multiplier

    def _preprocess(self, metrics: Any) -> List[float]:
        if isinstance(metrics, list):
            if self.metrics is not None:
                logger.warning(
                    f"Trying to get {self.metrics} keys from the metrics, "
                    "but the metric is a list. All values will be aggregated."
                )
            result = metrics
        elif isinstance(metrics, dict):
            if self.metrics is not None:
                if self.mode == "weighted_sum":
                    result = [
                        metrics[key] * value
                        for key, value in self.metrics.items()
                    ]
                else:
                    result = [metrics[key] for key in self.metrics]
            else:
                result = list(metrics.values())
        else:
            result = [metrics]

        return result

    def on_batch_end(self, state: "_State") -> None:
        """
        Computes the metric and add it to the metrics
        """
        metrics = self._preprocess(state.batch_metrics)
        metric = self.aggregation_fn(metrics)
        state.batch_metrics[self.prefix] = metric
