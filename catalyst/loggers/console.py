from typing import Dict, TYPE_CHECKING

from catalyst.core.logger import ILogger

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


def _format_metrics(dct: Dict):
    return " | ".join([f"{k}: {float(dct[k])}" for k in sorted(dct.keys())])


class ConsoleLogger(ILogger):
    """Console logger for parameters and metrics.
    Output the metric into the console during experiment.

    Args:
        log_hparams: boolean flag to print all hparams to the console (default: False)

    .. note::
        This logger is used by default by all Runners.
    """

    def __init__(self, log_hparams: bool = False):
        super().__init__(log_batch_metrics=False, log_epoch_metrics=True)
        self._log_hparams = log_hparams

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyperparameters to the console."""
        if self._log_hparams:
            print(f"Hparams: {hparams}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs loader and epoch metrics to stdout."""
        if scope == "loader":
            prefix = f"{runner.loader_key} ({runner.epoch_step}/{runner.num_epochs}) "
            msg = prefix + _format_metrics(metrics)
            print(msg)
        elif scope == "epoch":
            # @TODO: remove trick to save pure epoch-based metrics, like lr/momentum
            prefix = f"* Epoch ({runner.epoch_step}/{runner.num_epochs}) "
            msg = prefix + _format_metrics(metrics["_epoch_"])
            print(msg)


__all__ = ["ConsoleLogger"]
