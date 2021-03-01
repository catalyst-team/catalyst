import torch
from catalyst.dl import Callback, CallbackOrder, CallbackNode
from typing import Union, List, Dict, Any


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
            prefix: new key for aggregated metric.
            metrics (Union[str, List[str], Dict[str, float]]): If not None,
                it aggregates only the values from the metric by these keys.
                for ``weighted_sum`` aggregation it must be a Dict[str, float].
            mode: function for aggregation.
                Must be either ``sum``, ``mean`` or ``weighted_sum`` or user's
                function to aggregate metrics. This function must get dict of
                metrics and runner and return aggregated metric. It can be
                useful for complicated fine tuning with different losses that
                depends on epochs and loader or something also
            scope: type of metric. Must be either ``batch``, ``loader`` or
                ``epoch``
            multiplier: scale factor for the aggregated metric.
        Examples:
            Loss depends on epoch(Note that all loss functions that are used
            in the aggregation function must be defined.)
            >>> from catalyst.dl import MetricAggregationCallback
            >>>
            >>> def aggregation_function(metrics, runner):
            >>>     epoch = runner.stage_epoch_step
            >>>     region_loss = metrics["loss_dice"] + metrics["loss_iou"]
            >>>     bce_loss = metrics['loss_bce']
            >>>     loss = 1 / epoch * bce_loss + epoch / 3 * region_loss
            >>>     return loss
            >>>
            >>> MetricAggregationCallback(
            >>>     mode=aggregation_function,
            >>>     prefix='loss',
            >>> )
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
        elif not callable(mode):
            raise NotImplementedError(
                "mode must be `sum`, `mean` "
                "or `weighted_sum` or `weighted_mean` or be Callable"
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
        elif callable(mode):
            self.aggregation_fn = mode

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

    def _process_metrics(self, metrics: Dict, runner: "IRunner"):
        if callable(self.mode):
            metric_aggregated = self.aggregation_fn(metrics, runner)
        else:
            metrics_processed = self._preprocess(metrics)
            metric_aggregated = self.aggregation_fn(metrics_processed)
        metrics[self.prefix] = metric_aggregated

    def on_batch_end(self, runner: "IRunner") -> None:
        """Computes the metric and add it to the batch metrics.
        Args:
            runner: current runner
        """
        if self.scope == "batch":
            self._process_metrics(runner.batch_metrics, runner)

    def on_loader_end(self, runner: "IRunner"):
        """Computes the metric and add it to the loader metrics.
        Args:
            runner: current runner
        """
        self._process_metrics(runner.loader_metrics, runner)


__all__ = ["MetricAggregationCallback", ]