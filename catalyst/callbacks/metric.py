# @TODO: add metric aggregation, etc callback
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics.misc import ILoaderMetric, IMetric


class IMetricCallback(Callback):
    """Metric callback interface, abstraction over metric step."""

    pass


# @TODO: add KV support
class MetricCallback(IMetricCallback):
    def __init__(
        self, metric: IMetric, input_key: str, target_key: str, compute_on_batch: bool = True,
    ):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.metric = metric
        self.input_key = input_key
        self.target_key = target_key
        self.compute_on_batch = compute_on_batch

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)

        self.metric.update(inputs, targets)
        if self.compute_on_batch:
            runner.batch_metrics.update(self.metric.compute_key_value())

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics.update(self.metric.compute_key_value())


class LoaderMetricCallback(MetricCallback):
    def __init__(self, metric: ILoaderMetric, input_key: str, target_key: str):
        super().__init__(
            metric=metric, input_key=input_key, target_key=target_key, compute_on_batch=False,
        )

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset(batch_len=runner.loader_batch_len, sample_len=runner.loader_sample_len)


__all__ = ["IMetricCallback", "MetricCallback", "LoaderMetricCallback"]
