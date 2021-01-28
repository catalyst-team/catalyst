# @TODO: add metric aggregation, etc callback
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics.misc import IBatchMetric, ILoaderMetric


class IMetricCallback(Callback):
    """Metric callback interface, abstraction over metric step."""

    pass


# @TODO: add KV support
class MetricCallback(IMetricCallback):
    def __init__(
        self, metric: IBatchMetric, input_key: str, target_key: str, log_on_batch: bool = True,
    ):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        assert isinstance(metric, IBatchMetric)
        self.metric = metric
        self.input_key = input_key
        self.target_key = target_key
        self.log_on_batch = log_on_batch

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)

        metrics = self.metric.update_key_value(inputs, targets)
        if self.log_on_batch:
            runner.batch_metrics.update(metrics)

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics.update(self.metric.compute_key_value())


class LoaderMetricCallback(IMetricCallback):
    def __init__(self, metric: ILoaderMetric, input_key: str, target_key: str):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        assert isinstance(metric, ILoaderMetric)
        self.metric = metric
        self.input_key = input_key
        self.target_key = target_key

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset(
            num_batches=runner.loader_batch_len, num_samples=runner.loader_sample_len
        )

    def on_batch_end(self, runner: "IRunner") -> None:
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)
        self.metric.update(inputs, targets)

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics.update(self.metric.compute_key_value())


__all__ = ["IMetricCallback", "MetricCallback", "LoaderMetricCallback"]
