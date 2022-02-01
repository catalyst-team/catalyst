from typing import Dict

from catalyst.core.logger import ILogger


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

    def log_hparams(self, hparams: Dict) -> None:
        """Logs hyperparameters to the console."""
        if self._log_hparams:
            print(f"Hparams: {hparams}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        num_epochs: int = 0,
        epoch_step: int = 0,
        batch_step: int = 0,
        sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs loader and epoch metrics to stdout."""
        if scope == "loader":
            prefix = f"{loader_key} ({epoch_step}/{num_epochs}) "
            msg = prefix + _format_metrics(metrics)
            print(msg)
        elif scope == "epoch":
            # @TODO: remove trick to save pure epoch-based metrics, like lr/momentum
            prefix = f"* Epoch ({epoch_step}/{num_epochs}) "
            msg = prefix + _format_metrics(metrics["_epoch_"])
            print(msg)


__all__ = ["ConsoleLogger"]
