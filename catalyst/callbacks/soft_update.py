from catalyst.core import CallbackOrder, IRunner
from catalyst.core.callback import Callback
from catalyst.utils.torch import soft_update


class SoftUpdateCallaback(Callback):
    """Callback to update `target` data inside `runner.model` with the `source`
    data inside `runner.model` one smoothing by ``tau`` (inplace operation).

    Args:
        target_model: key to the data inside `runner.model` to update
        source_model: key to the source data inside `runner.model`
        tau: smoothing parameter `target * (1.0 - tau) + source * tau`
        scope (str): when the `target` should be updated
                ``"on_batch_end"``
                ``"on_batch_start"``
                ``"on_epoch_end"``
                ``"on_epoch_start"``

    Raises:
        TypeError: if invalid scope
    """

    def __init__(
        self, target_model: str, source_model: str, tau: float, scope: str
    ) -> None:
        """Init."""
        super().__init__(order=CallbackOrder.External)
        self.target_model = target_model
        self.source_model = source_model
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

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        assert self.target_model in runner.model, (
            f"Could not find speficied target model ({self.target_model}) "
            "within available runner models ({runner.model.keys()})"
        )
        assert self.source_model in runner.model, (
            f"Could not find speficied target model ({self.source_model}) "
            "within available runner models ({runner.model.keys()})"
        )

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader and self.scope == "on_epoch_start":
            soft_update(
                runner.model[self.target_model],
                runner.model[self.source_model],
                self.tau,
            )

    def on_batch_start(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader and self.scope == "on_batch_start":
            soft_update(
                runner.model[self.target_model],
                runner.model[self.source_model],
                self.tau,
            )

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader and self.scope == "on_batch_end":
            soft_update(
                runner.model[self.target_model],
                runner.model[self.source_model],
                self.tau,
            )

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader and self.scope == "on_epoch_end":
            soft_update(
                runner.model[self.target_model],
                runner.model[self.source_model],
                self.tau,
            )


__all__ = ["SoftUpdateCallaback"]
