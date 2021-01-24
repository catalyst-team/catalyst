from typing import Any, Dict
import os

import numpy as np

from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core.logger import ILogger
from catalyst.utils.config import save_config


def _format_metrics(dct: Dict):
    return " | ".join([f"{k}: {dct[k]}" for k in sorted(dct.keys())])


class ConsoleLogger(ILogger):
    # def __init__(self, include: List[str] = None, exclude: List[str] = None):
    #     self.include = include
    #     self.exclude = exclude

    def log_metrics(
        self,
        metrics: Dict[str, float],
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
        # if self.exclude is not None and scope in self.exclude:
        #     return
        # elif (
        #     self.include is not None and scope in self.include
        # ) or self.include is None:
        if scope == "loader":
            prefix = f"{loader_key} ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics)
            print(msg)
        elif scope == "epoch":
            # @TODO: trick to save pure epoch-based metrics, like lr/momentum
            prefix = f"* Epoch ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics["_epoch_"])
            print(msg)

    def log_image(
        self,
        image: np.ndarray,
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
        pass

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        experiment_key: str = None,
    ) -> None:
        print(f"Hparams ({experiment_key}): {hparams}")


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


class TensorboardLogger(ILogger):
    """Logger callback, translates ``runner.metric_manager`` to tensorboard."""

    def __init__(self, logdir: str):
        self.logdir = logdir
        self.loggers = {}
        os.makedirs(self.logdir, exist_ok=True)

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            self.loggers[loader_key].add_scalar(f"{key}{suffix}", value, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
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
        if scope == "batch":
            if loader_key not in self.loggers.keys():
                logdir = os.path.join(self.logdir, f"{loader_key}")
                self.loggers[loader_key] = SummaryWriter(logdir)
            self._log_metrics(
                metrics=metrics, step=stage_batch_step, loader_key=loader_key, suffix="/batch"
            )
        elif scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                if loader_key not in self.loggers.keys():
                    logdir = os.path.join(self.logdir, f"{loader_key}")
                    self.loggers[loader_key] = SummaryWriter(logdir)
                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=stage_epoch_step,
                    loader_key=loader_key,
                    suffix="/epoch",
                )

    def flush_log(self) -> None:
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        for logger in self.loggers.values():
            logger.close()


__all__ = ["ConsoleLogger", "CSVLogger", "TensorboardLogger"]
