from typing import Dict, TYPE_CHECKING
import os

from catalyst.core.logger import ILogger
from catalyst.utils.config import save_config

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class CSVLogger(ILogger):
    """CSV logger for the metrics storing under ``.csv`` file.

    Args:
        logdir: path to logdir for the logger
        use_logdir_postfix: boolean flag to use extra ``logs`` prefix in the logdir

    .. note::
        This logger is used by default by ``dl.Runner`` and ``dl.SupervisedRunner``
        in case of specified logdir during ``runner.train(..., logdir=/path/to/logdir)``.

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"csv": dl.CSVLogger(logdir="./logdir/logs"}
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "csv": dl.CSVLogger(logdir="./logdir/logs")
                }

            # ...

        runner = CustomRunner().run()

    """

    def __init__(self, logdir: str, use_logdir_postfix: bool = False):
        """Init."""
        super().__init__(log_batch_metrics=False, log_epoch_metrics=True)
        if use_logdir_postfix:
            logdir = os.path.join(logdir, "csv_logger")
        self.logdir = logdir
        self.loggers = {}
        os.makedirs(self.logdir, exist_ok=True)

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.loggers

    def _make_header(self, metrics: Dict[str, float], loader_key: str):
        log_line_header = "step,"
        for metric in sorted(metrics.keys()):
            log_line_header += metric + ","
        log_line_header = log_line_header[:-1] + "\n"  # replace last "," with new line
        self.loggers[loader_key].write(log_line_header)

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str):
        log_line_csv = f"{step},"
        for metric in sorted(metrics.keys()):
            log_line_csv += str(metrics[metric]) + ","
        log_line_csv = log_line_csv[:-1] + "\n"  # replace last "," with new line
        self.loggers[loader_key].write(log_line_csv)

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyperparameters to the logger."""
        save_config(config=hparams, path=os.path.join(self.logdir, "hparams.json"))

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs epoch metrics to csv file."""
        if scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                if loader_key not in self.loggers.keys():
                    self.loggers[loader_key] = open(
                        os.path.join(self.logdir, f"{loader_key}.csv"), "a+"
                    )
                    self._make_header(metrics=per_loader_metrics, loader_key=loader_key)
                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=runner.epoch_step,
                    loader_key=loader_key,
                )

    def flush_log(self) -> None:
        """Flushes the logger."""
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        """Closes the logger."""
        for logger in self.loggers.values():
            logger.close()


__all__ = ["CSVLogger"]
