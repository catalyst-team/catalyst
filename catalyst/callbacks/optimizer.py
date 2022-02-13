from typing import Callable, Dict, TYPE_CHECKING, Union
from functools import partial

from catalyst.core.callback import IOptimizerCallback
from catalyst.registry import REGISTRY
from catalyst.utils.misc import get_attr
from catalyst.utils.torch import get_optimizer_momentum_list

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class OptimizerCallback(IOptimizerCallback):
    """Optimizer callback, abstraction over optimizer step.

    Args:
        metric_key: a key to get loss from ``runner.batch_metrics``
        model_key: a key to select a model from ``runner.model`` in case
            there are several of them and they are in a dictionary format.
        optimizer_key: a key to select a optimizer from ``runner.optimizer`` in case
            there are several of them and they are in a dictionary format.
        accumulation_steps: number of steps before ``optimizer.step()``
        grad_clip_fn: callable gradient cliping function or it's name or
        grad_clip_params: key-value parameters for grad_clip_fn

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        metric_key: str,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_fn: Union[str, Callable] = None,
        grad_clip_params: Dict = None,
    ):
        """Init."""
        super().__init__()
        self.metric_key = metric_key
        self.optimizer_key = optimizer_key
        self.optimizer = None
        self.criterion = None

        if isinstance(grad_clip_fn, str):
            self.grad_clip_fn = REGISTRY.get(grad_clip_fn)
        else:
            self.grad_clip_fn = grad_clip_fn
        if grad_clip_params is not None:
            self.grad_clip_fn = partial(self.grad_clip_fn, **grad_clip_params)

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        if self.optimizer_key is not None:
            self._prefix = f"{self.optimizer_key}"
            self._prefix_lr = f"lr/{self._prefix}"
            self._prefix_momentum = f"momentum/{self._prefix}"
            self._prefix_gradient = f"gradient/{self._prefix}"
        else:
            self._prefix_lr = "lr"
            self._prefix_momentum = "momentum"
            self._prefix_gradient = "gradient"

    def _get_lr_momentum_stats(self) -> Dict:
        lr_list = [param_group["lr"] for param_group in self.optimizer.param_groups]
        momentum_list = get_optimizer_momentum_list(self.optimizer)
        stats = {self._prefix_lr: lr_list[0], self._prefix_momentum: momentum_list[0]}
        return stats

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.optimizer = get_attr(runner, key="optimizer", inner_key=self.optimizer_key)
        assert self.optimizer is not None

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        if runner.is_train_loader:
            self._accumulation_counter += 1
            need_gradient_step = (
                self._accumulation_counter % self.accumulation_steps == 0
            )

            if need_gradient_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

        runner.batch_metrics.update(self._get_lr_momentum_stats())

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler."""
        runner.loader_metrics.update(self._get_lr_momentum_stats())


__all__ = ["OptimizerCallback"]
