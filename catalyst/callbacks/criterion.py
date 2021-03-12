import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics._additive import AdditiveValueMetric
from catalyst.utils.misc import get_attr


class ICriterionCallback(Callback):
    """Criterion callback interface, abstraction over criterion step."""

    pass


# @TODO: add KV support
class CriterionCallback(ICriterionCallback):
    """Criterion callback, abstraction over criterion step.

    Args:
        input_key:
        target_key:
        metric_key: prefix for metrics and output key for loss
            in ``runner.batch_metrics`` dictionary
        criterion_key: A key to take a criterion in case
            there are several of them and they are in a dictionary format.
    """

    def __init__(
        self, input_key: str, target_key: str, metric_key: str, criterion_key: str = None,
    ):
        """Init."""
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.input_key = input_key
        self.target_key = target_key
        self.metric_key = metric_key
        self.criterion_key = criterion_key
        self.additive_metric = AdditiveValueMetric()
        self.criterion = None

    def on_stage_start(self, runner: "IRunner"):
        """Checks that the current stage has correct criterion.

        Args:
            runner: current runner
        """
        self.criterion = get_attr(runner, key="criterion", inner_key=self.criterion_key)
        assert self.criterion is not None

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.additive_metric.reset()

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]

        # NOTE: similar to amp guides in docs
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        # with runner.engine.autocast():
        loss = self.criterion(inputs, targets)

        runner.batch_metrics.update({self.metric_key: loss})
        self.additive_metric.update(loss.detach().cpu(), len(targets))

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler."""
        mean, std = self.additive_metric.compute()
        metrics = {self.metric_key: mean, f"{self.metric_key}/std": std}
        metrics = {
            k: runner.engine.sync_tensor(torch.tensor(v, device=runner.device), "mean")
            for k, v in metrics.items()
        }
        runner.loader_metrics.update(metrics)


__all__ = ["ICriterionCallback", "CriterionCallback"]
