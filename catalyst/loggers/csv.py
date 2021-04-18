from typing import Any, Dict
import os

from catalyst.core.logger import ILogger
from catalyst.utils.config import save_config


class CSVLogger(ILogger):
    """CSV logger for the metrics storing under ``.csv`` file.

    Args:
        logdir: path to logdir for the logger
        use_logdir_postfix: boolean flag to use extra ``logs`` prefix in the logdir

    .. note::
        This logger is used by default by ``dl.Runner`` and ``dl.SupervisedRunner`` in case of
        specified logdir during ``runner.train(..., logdir=/path/to/logdir)``.

    .. note::
        This logger is used by default by ``dl.ConfigRunner`` and ``dl.HydraRunner`` in case of
        specified logdir in config ``args``.

    Notebook API examples:

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

    Config API example:

    .. code-block:: yaml

        loggers:
            csv:
                _target_: CSVLogger
                logdir: ./logdir/logs
        ...

    Hydra API example:

    .. code-block:: yaml

        loggers:
            csv:
                _target_: catalyst.dl.CSVLogger
                logdir: ./logdir/logs
        ...
    """

    def __init__(self, logdir: str, use_logdir_postfix: bool = False):
        """Init."""
        if use_logdir_postfix:
            logdir = os.path.join(logdir, "logs")
        self.logdir = logdir
        self.loggers = {}
        os.makedirs(self.logdir, exist_ok=True)

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

    def log_metrics(
        self,
        metrics: Dict[str, Any],
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
        """@TODO: docs."""
        if scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                if loader_key not in self.loggers.keys():
                    self.loggers[loader_key] = open(
                        os.path.join(self.logdir, f"{loader_key}.csv"), "a+"
                    )
                    self._make_header(metrics=per_loader_metrics, loader_key=loader_key)
                self._log_metrics(
                    metrics=per_loader_metrics, step=stage_epoch_step, loader_key=loader_key,
                )

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """@TODO: docs."""
        if scope == "experiment":
            save_config(config=hparams, path=os.path.join(self.logdir, "hparams.yml"))

    def flush_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.close()


__all__ = ["CSVLogger"]
