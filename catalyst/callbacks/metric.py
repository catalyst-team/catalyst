# @TODO: add metric aggregation, etc callback
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics._metric import ICallbackBatchMetric, ICallbackLoaderMetric


class IMetricCallback(Callback):
    """Metric callback interface, abstraction over metric step."""

    pass


# @TODO: add KV support for input/output
class BatchMetricCallback(IMetricCallback):
    """@TODO: docs."""

    def __init__(
        self,
        metric: ICallbackBatchMetric,
        input_key: str,
        target_key: str,
        log_on_batch: bool = True,
    ):
        """@TODO: docs."""
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        assert isinstance(metric, ICallbackBatchMetric)
        self.metric = metric
        self.input_key = input_key
        self.target_key = target_key
        self.log_on_batch = log_on_batch

    def on_loader_start(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)

        metrics = self.metric.update_key_value(inputs, targets)
        if self.log_on_batch:
            runner.batch_metrics.update(metrics)

    def on_loader_end(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        runner.loader_metrics.update(self.metric.compute_key_value())


class LoaderMetricCallback(IMetricCallback):
    """@TODO: docs."""

    def __init__(self, metric: ICallbackLoaderMetric, input_key: str, target_key: str):
        """@TODO: docs."""
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        assert isinstance(metric, ICallbackLoaderMetric)
        self.metric = metric
        self.input_key = input_key
        self.target_key = target_key

    def on_loader_start(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        self.metric.reset(
            num_batches=runner.loader_batch_len, num_samples=runner.loader_sample_len
        )

    def on_batch_end(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)
        self.metric.update(inputs, targets)

    def on_loader_end(self, runner: "IRunner") -> None:
        """@TODO: docs."""
        runner.loader_metrics.update(self.metric.compute_key_value())


__all__ = ["IMetricCallback", "BatchMetricCallback", "LoaderMetricCallback"]
