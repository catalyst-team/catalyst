from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics.misc import AverageMetric
from catalyst.utils.misc import get_attr


class ICriterionCallback(Callback):
    """Criterion callback interface, abstraction over criterion step."""

    pass


# @TODO: add KV support
class CriterionCallback(ICriterionCallback):
    def __init__(
        self,
        input_key: str = None,
        target_key: str = None,
        metric_key: str = None,
        criterion_key: str = None,
    ):
        """
        Args:
            input_key:
            target_key:
            metric_key: prefix for metrics and output key for loss
                in ``runner.batch_metrics`` dictionary
            criterion_key: A key to take a criterion in case
                there are several of them and they are in a dictionary format.
        """
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.input_key = input_key
        self.target_key = target_key
        self.metric_key = metric_key
        self.criterion_key = criterion_key
        self.average_metric = AverageMetric()
        self.criterion = None

    def on_stage_start(self, runner: "IRunner"):
        """Checks that the current stage has correct criterion.

        Args:
            runner: current runner
        """
        self.criterion = get_attr(runner, key="criterion", inner_key=self.criterion_key)
        assert self.criterion is not None

    def on_loader_start(self, runner: "IRunner") -> None:
        self.average_metric.reset()

    def on_batch_end(self, runner: "IRunner"):
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)

        loss = self.criterion(inputs, targets)
        runner.batch_metrics.update({self.metric_key: loss})
        self.average_metric.update(loss.item(), len(targets))

    def on_loader_end(self, runner: "IRunner") -> None:
        mean, std = self.average_metric.compute()
        runner.loader_metrics.update({self.metric_key: mean, f"{self.metric_key}/std": std})


__all__ = ["ICriterionCallback", "CriterionCallback"]
