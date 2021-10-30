from catalyst.core import CallbackOrder, IRunner
from catalyst.core.callback import Callback
from catalyst.utils.torch import soft_update


class SoftUpdateCallaback(Callback):
    """Callback to update `target` data inside `runner.model` with the `source`
    data inside `runner.model` one smoothing by ``tau`` (inplace operation).

    Args:
        target_model_key: key to the data inside `runner.model` to update
        source_model_key: key to the source data inside `runner.model`
        tau: smoothing parameter `target * (1.0 - tau) + source * tau`
        scope (str): when the `target` should be updated
                ``"on_batch_end"``
                ``"on_batch_start"``
                ``"on_epoch_end"``
                ``"on_epoch_start"``
    """

    def __init__(
        self, target_model_key: str, source_model_key: str, tau: float, scope: str
    ) -> None:
        """Init.

        Args:
            target_model_key: key to the data inside `runner.model` to update
            source_model_key: key to the source data inside `runner.model`
            tau: smoothing parameter `target * (1.0 - tau) + source * tau`
            scope (str): when the `target` should be updated
                ``"on_batch_end"``
                ``"on_batch_start"``
                ``"on_epoch_end"``
                ``"on_epoch_start"``

        Raises:
            TypeError: if invalid scope
        """
        super().__init__(order=CallbackOrder.Metric)
        self.target_model_key = target_model_key
        self.source_model_key = source_model_key
        self.tau = tau
        if isinstance(scope, str) and scope in [
            "on_batch_end",
            "on_batch_start",
            "on_epoch_end",
            "on_epoch_start",
        ]:
            self.scope = scope
        else:
            raise TypeError(
                """Expected scope to be on of the: [
                    "on_batch_end",
                    "on_batch_start",
                    "on_epoch_end",
                    "on_epoch_start"]"""
            )

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action.

        Args:
            runner: runner for the experiment.
        """
        if runner.is_train_loader and self.scope == "on_batch_end":
            soft_update(
                runner.model[self.target_model_key], runner.model[self.source_model_key], self.tau
            )

    def on_batch_start(self, runner: "IRunner") -> None:
        """On batch start action.

        Args:
            runner: runner for the experiment.
        """
        if runner.is_train_loader and self.scope == "on_batch_start":
            soft_update(
                runner.model[self.target_model_key], runner.model[self.source_model_key], self.tau
            )

    def on_epoch_end(self, runner: "IRunner") -> None:
        """On epoch end action.

        Args:
            runner: runner for the experiment.
        """
        if runner.is_train_loader and self.scope == "on_epoch_end":
            soft_update(
                runner.model[self.target_model_key], runner.model[self.source_model_key], self.tau
            )

    def on_epoch_start(self, runner: "IRunner") -> None:
        """On epoch start action.

        Args:
            runner: runner for the experiment.
        """
        if runner.is_train_loader and self.scope == "on_epoch_start":
            soft_update(
                runner.model[self.target_model_key], runner.model[self.source_model_key], self.tau
            )


__all__ = ["SoftUpdateCallaback"]
