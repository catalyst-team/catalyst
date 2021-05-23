from typing import Dict

from catalyst.core.logger import ILogger


def _format_metrics(dct: Dict):
    return " | ".join([f"{k}: {float(dct[k])}" for k in sorted(dct.keys())])


class ConsoleLogger(ILogger):
    """Console logger for parameters and metrics.
    Output the metric into the console during experiment.

    Args:
        log_hparams: boolean flag to print all hparams to the console (default: False)
        log_loader_metrics: boolean flag to print loader metrics to the console (default: True)
        log_epoch_metrics: boolean flag to print epoch metrics to the console (default: True)

    .. note::
        This logger is used by default by all Runners.
    """

    def __init__(
        self,
        log_hparams: bool = False,
        log_loader_metrics: bool = True,
        log_epoch_metrics: bool = True,
    ):
        super().__init__()
        self._log_hparams = log_hparams
        self._log_loader_metrics = log_loader_metrics
        self._log_epoch_metrics = log_epoch_metrics

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs loader and epoch metrics to stdout."""
        if scope == "loader" and self._log_loader_metrics:
            prefix = f"{loader_key} ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics)
            print(msg)
        elif scope == "epoch" and self._log_epoch_metrics:
            # @TODO: trick to save pure epoch-based metrics, like lr/momentum
            prefix = f"* Epoch ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics["_epoch_"])
            print(msg)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the console.

        Args:
            hparams: Parameters to log.
            scope: On which scope log parameters.
            run_key: Experiment info.
            stage_key: Stage info.
        """
        if scope == "experiment" and self._log_hparams:
            print(f"Hparams ({run_key}): {hparams}")


__all__ = ["ConsoleLogger"]
