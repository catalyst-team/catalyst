from typing import Any, Dict
import os

from catalyst.core.logger import ILogger
from catalyst.utils.config import save_config


class CSVLogger(ILogger):
    def __init__(self, logdir: str):
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
        experiment_key: str = None,
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
        experiment_key: str = None,
    ) -> None:
        save_config(config=hparams, path=os.path.join(self.logdir, "hparams.yml"))

    def flush_log(self) -> None:
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        for logger in self.loggers.values():
            logger.close()


__all__ = ["CSVLogger"]
