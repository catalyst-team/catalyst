import os
import logging
import json
from typing import List, Dict

from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from .base import LoggerCallback
from .utils import to_batch_metrics


class TxtMetricsFormatter(logging.Formatter):
    """
    Translate batch metrics in human-readable format.

    This class is used by logging.Logger to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        fmt = "[{asctime}] {message}"
        super().__init__(fmt, style="{")

    @staticmethod
    def _get_metrics_string(metrics):
        return " | ".join(
            "{}: {:.5f}".format(k, v) for k, v in sorted(metrics.items())
        )

    @staticmethod
    def _format_metrics_message(state):
        message = f"{state.epoch} * Epoch metrics:\n"
        for k, v in sorted(state.epoch_metrics.items()):
            message += f"({k}) {TxtMetricsFormatter._get_metrics_string(v)}\n"
        return message

    def format(self, record):
        record.msg = self._format_metrics_message(record.state)
        return super().format(record)


class JsonMetricsFormatter(logging.Formatter):
    """
    Translate batch metrics in json format.

    This class is used by logging.Logger to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        fmt = "{message}"
        super().__init__(fmt, style="{")

    def format(self, record):
        state = record.state
        dct = {}
        dct["epoch_metrics"] = state.epoch_metrics.copy()
        dct["epoch"] = state.epoch
        dct["asctime"] = self.formatTime(record)
        return json.dumps(dct)


class Logger(LoggerCallback):
    """
    Logger callback, translates state.*_metrics to console and text file
    """

    def __init__(self, logdir: str = None):
        """
        :param logdir: log directory to use for text logging
        """
        super().__init__(logdir)
        self.logger = None

    def on_train_start(self, state):
        super().on_train_start(state)
        logger_name = os.path.join(self.logdir, "logs")
        self.logger = self._get_logger(logger_name)

    @staticmethod
    def _get_logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(logger_name + ".txt")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        jh = logging.FileHandler(logger_name + ".json")
        jh.setLevel(logging.INFO)

        txt_formatter = TxtMetricsFormatter()
        json_formatter = JsonMetricsFormatter()
        fh.setFormatter(txt_formatter)
        ch.setFormatter(txt_formatter)
        jh.setFormatter(json_formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.addHandler(jh)
        return logger

    def on_epoch_end(self, state):
        if self.logger is not None:
            self.logger.info("", extra={"state": state})


class TensorboardLogger(LoggerCallback):
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
        super().__init__(logdir)
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
            self.metrics_to_log = sorted(list(metrics.keys()))

        for name in self.metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_batch_end(self, state: RunnerState):
        if self.log_on_batch_end:
            mode = state.loader_mode

            to_batch_metrics(state=state, metric_key="base/lr", state_key="lr")
            to_batch_metrics(
                state=state, metric_key="base/momentum", state_key="momentum")
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
