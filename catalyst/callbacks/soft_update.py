from torch import nn

from catalyst.core import CallbackOrder, IRunner
from catalyst.core.callback import Callback


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Updates the `target` data with the `source` one smoothing by ``tau`` (inplace operation)."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SoftUpdateCallaback(Callback):
    def __init__(self, target_model_key: str, source_model_key: str, tau: float) -> None:
        """Init."""
        super().__init__(order=CallbackOrder.Metric)
        self.target_model_key = target_model_key
        self.source_model_key = source_model_key
        self.tau = tau

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action.

        Args:
            runner: runner for the experiment.
        """
        soft_update(
            runner.model[self.target_model_key], runner.model[self.source_model_key], self.tau
        )


__all__ = ["SoftUpdateCallaback"]
