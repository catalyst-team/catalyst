from typing import Dict, TYPE_CHECKING
import logging

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.typing import Optimizer
from catalyst.utils import get_optimizer_momentum_list
from catalyst.utils.misc import get_attr

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

logger = logging.getLogger(__name__)


def zero_grad(optimizer: Optimizer) -> None:
    """Perform an hacky way to zero gradients.

    Args:
        optimizer: optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None


class IOptimizerCallback(Callback):
    """Optimizer callback interface, abstraction over optimizer step."""

    pass


class OptimizerCallback(IOptimizerCallback):
    """Optimizer callback, abstraction over optimizer step."""

    def __init__(
        self,
        metric_key: str,
        model_key: str = None,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
    ):
        """
        Args:
            metric_key: a key to get loss from ``runner.batch_metrics``
            model_key: a key to select a model from ``runner.model`` in case
                there are several of them and they are in a dictionary format.
            optimizer_key: a key to select a optimizer from ``runner.optimizer`` in case
                there are several of them and they are in a dictionary format.
            accumulation_steps: number of steps before ``model.zero_grad()``
        """
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        self.metric_key = metric_key
        self.model_key = model_key
        self.optimizer_key = optimizer_key
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        if self.model_key is not None or self.optimizer_key is not None:
            if self.model_key is not None and self.optimizer_key is not None:
                self._prefix = f"{self.model_key}_{self.optimizer_key}"
            elif self.model_key is not None:
                self._prefix = f"{self.model_key}"
            elif self.optimizer_key is not None:
                self._prefix = f"{self.optimizer_key}"
            self._prefix_lr = f"lr/{self._prefix}"
            self._prefix_momentum = f"momentum/{self._prefix}"
        else:
            self._prefix_lr = "lr"
            self._prefix_momentum = "momentum"

    def _get_lr_momentum_stats(self) -> Dict:
        lr_list = [param_group["lr"] for param_group in self.optimizer.param_groups]
        momentum_list = get_optimizer_momentum_list(self.optimizer)
        stats = {self._prefix_lr: lr_list[0], self._prefix_momentum: momentum_list[0]}
        return stats

    def on_stage_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.model = get_attr(runner, key="model", inner_key=self.model_key)
        self.optimizer = get_attr(runner, key="optimizer", inner_key=self.optimizer_key)
        assert self.model is not None
        assert self.optimizer is not None

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        if runner.is_train_loader:
            self._accumulation_counter += 1
            need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

            loss = runner.batch_metrics[self.metric_key]
            runner.engine.backward_loss(loss, self.model, self.optimizer)

            if need_gradient_step:
                runner.engine.optimizer_step(loss, self.model, self.optimizer)
                runner.engine.zero_grad(loss, self.model, self.optimizer)

        runner.batch_metrics.update(self._get_lr_momentum_stats())

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler."""
        runner.loader_metrics.update(self._get_lr_momentum_stats())


__all__ = [
    "IOptimizerCallback",
    "OptimizerCallback",
]
