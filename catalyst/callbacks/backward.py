from typing import Callable, Dict, TYPE_CHECKING, Union
from functools import partial

from catalyst.core.callback import IBackwardCallback
from catalyst.registry import REGISTRY

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class BackwardCallback(IBackwardCallback):
    """Optimizer callback, abstraction over backward step.

    Args:
        metric_key: a key to get loss from ``runner.batch_metrics``
        grad_clip_fn: callable gradient cliping function or it's name
        grad_clip_params: key-value parameters for grad_clip_fn
        log_gradient: boolean flag to log gradient norm to ``runner.batch_metrics``

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        metric_key: str,
        grad_clip_fn: Union[str, Callable] = None,
        grad_clip_params: Dict = None,
        log_gradient: bool = False,
    ):
        """Init."""
        super().__init__()
        self.metric_key = metric_key

        if isinstance(grad_clip_fn, str):
            self.grad_clip_fn = REGISTRY.get(grad_clip_fn)
        else:
            self.grad_clip_fn = grad_clip_fn
        if grad_clip_params is not None:
            self.grad_clip_fn = partial(self.grad_clip_fn, **grad_clip_params)

        self._prefix_gradient = f"gradient/{metric_key}"
        self._log_gradient = log_gradient

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        if runner.is_train_loader:
            loss = runner.batch_metrics[self.metric_key]
            runner.engine.backward(loss)

            if self.grad_clip_fn is not None:
                runner.engine.unscale_gradients()
                norm = self.grad_clip_fn(self.model.parameters())
                if self._log_gradient:
                    runner.batch_metrics[f"{self._prefix_gradient}/norm"] = norm


__all__ = ["BackwardCallback"]
