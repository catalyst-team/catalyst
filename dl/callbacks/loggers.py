import os
import logging
from typing import List, Dict

from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from .core import Callback
from .utils import to_batch_metrics


class Logger(Callback):
    """
    Logger callback, translates state.*_metrics to console and text file
    """

    def __init__(self, logdir: str = None):
        """
        :param logdir: log directory to use for text logging
        """
        self.logger = None
        self._logdir = logdir

    @property
    def logdir(self):
        return self._logdir

    @logdir.setter
    def logdir(self, value):
        self._logdir = value
        os.makedirs(value, exist_ok=True)
        log_filepath = os.path.join(value, "logs.txt")
        self.logger = self._get_logger(log_filepath)

    @staticmethod
    def _get_logger(log_filepath):
        logger = logging.getLogger(log_filepath)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def _get_metrics_string(metrics):
        return " | ".join(
            "{}: {:.5f}".format(k, v) for k, v in metrics.items()
        )

    def on_train_begin(self, state):
        if self.logger is not None:
            self.logger.info(
                "Starting training with params:\n{}\n\n".format(state)
            )

    def on_epoch_end(self, state):
        if self.logger is not None:
            for k, v in state.epoch_metrics.items():
                self.logger.info(
                    f"{state.epoch} * Epoch ({k}) metrics: "
                    f"{self._get_metrics_string(v)}"
                )
            self.logger.info("\n")


class TensorboardLogger(Callback):
    """
    Logger callback, translates state.*_metrics to tensorboard
    """

    def __init__(
        self,
        logdir: str = None,
        metric_names: List[str] = None,
        log_on_batch_end=True,
        log_on_epoch_end=True
    ):
        """
        :param logdir: directory where logs will be created
        :param metric_names: List of metric names to log.
            If none - logs everything.
        :param log_on_batch_end: Logs per-batch value of metrics,
            prepends 'batch_' prefix to their names.
        :param log_on_epoch_end: Logs per-epoch metrics if set True.
        """
        self.logdir = logdir
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        # You definitely should log something)
        assert self.log_on_batch_end or self.log_on_epoch_end
        self.loggers = dict()

    def on_loader_start(self, state):
        lm = state.loader_mode
        if lm not in self.loggers:
            self.loggers[lm] = UtilsFactory.create_tflogger(
                logdir=self.logdir, name=lm
            )

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            self.metrics_to_log = list(metrics.keys())

        for name in self.metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_batch_end(self, state: RunnerState):
        if self.log_on_batch_end:
            mode = state.loader_mode

            to_batch_metrics(state=state, metric_key="lr")
            to_batch_metrics(state=state, metric_key="momentum")
            to_batch_metrics(state=state, metric_key="loss")

            self._log_metrics(
                metrics=state.batch_metrics,
                step=state.step,
                mode=mode,
                suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        if self.log_on_epoch_end:
            mode = state.loader_mode
            self._log_metrics(
                metrics=state.epoch_metrics[mode],
                step=state.epoch,
                mode=mode,
                suffix="/epoch"
            )
