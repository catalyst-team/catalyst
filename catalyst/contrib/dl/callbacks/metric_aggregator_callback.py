from typing import Any, List, Dict, Union  # isort:skip
import logging

from catalyst.dl import Callback, CallbackOrder, State

logger = logging.getLogger(__name__)


class MetricAggregatorCallback(Callback):
    """
    A callback to aggregate several metrics in one value.
    """
    def __init__(
        self,
        prefix: str,
        metric_keys: Union[str, List[str], Dict[str, float]] = None,
        metric_aggregate_fn: str = "sum",
        multiplier: float = 1.0
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated metric.
            metric_keys (Union[str, List[str], Dict[str, float]]): If not None,
                it aggregates only the values from the metric by these keys.
                for ``weighted_sum`` aggregation it must be a Dict[str, float].
            metric_aggregate_fn (str): function for aggregation.
                Must be either ``sum``, ``mean`` or ``weighted_sum``.
            multiplier (float): scale factor for the aggregated metric.
        """
        super().__init__(CallbackOrder.Metric + 1)
        if prefix is None or not isinstance(prefix, str):
            raise ValueError("prefix must be str")
        self.prefix = prefix

        if isinstance(metric_keys, str):
            metric_keys = [metric_keys]
        self.metric_keys = metric_keys

        self.multiplier = multiplier

        if metric_aggregate_fn == "sum":
            self.metric_fn = lambda x: sum(x) * multiplier
        elif metric_aggregate_fn == "weighted_sum":
            if metric_keys is None or not isinstance(metric_keys, dict):
                raise ValueError(
                    "For `weighted_sum` mode the metric_keys must be specified"
                    " and must be a dict"
                )
            self.metric_fn = lambda x: sum(x) * multiplier
        elif metric_aggregate_fn == "mean":
            self.metric_fn = lambda x: sum(x) / len(x) * multiplier
        else:
            raise ValueError(
                "metric_aggregate_fn must be `sum`, `mean` or weighted_sum`"
            )

        self.metric_aggregate_name = metric_aggregate_fn

    def _preprocess(self, metrics: Any) -> List[float]:
        if isinstance(metrics, list):
            if self.metric_keys is not None:
                logger.warning(
                    f"Trying to get {self.metric_keys} keys from the metrics, "
                    "but the metric is a list. All values will be aggregated."
                )
            result = metrics
        elif isinstance(metrics, dict):
            if self.metric_keys is not None:
                if self.metric_aggregate_name == "weighted_sum":
                    result = [
                        metrics[key] * value
                        for key, value in self.metric_keys.items()
                    ]
                else:
                    result = [metrics[key] for key in self.metric_keys]
            else:
                result = list(metrics.values())
        else:
            result = [metrics]

        return result

    def on_batch_end(self, state: State) -> None:
        """
        Computes the metric and add it to the metrics
        """
        metrics = self._preprocess(state.metric_manager.batch_values)
        metric = self.metric_fn(metrics)

        state.metric_manager.add_batch_value(
            metrics_dict={
                self.prefix: metric,
            }
        )
