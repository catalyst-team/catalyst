from abc import ABC, abstractmethod
from typing import List, Dict
import sys
import logging
import json
from datetime import datetime
from tqdm import tqdm

from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory


class VerboseLogger(Callback):
    def __init__(self):
        self.tqdm: tqdm = None
        self.step = 0

    def on_loader_start(self, state: RunnerState):
        self.step = 0
        self.tqdm = tqdm(
            total=state.loader_len,
            desc=f"{state.stage_epoch}/{state.num_epochs}"
            f" * Epoch ({state.loader_name})",
            leave=True,
            ncols=0,
            file=sys.stdout
        )

    def on_batch_end(self, state: RunnerState):
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v)
                for k, v in
                sorted(state.metrics.batch_values.items())
                if not k.startswith("base")
            }
        )
        self.tqdm.update()

    def on_loader_end(self, state: RunnerState):
        self.tqdm.close()
        self.tqdm = None
        self.step = 0


class MetricsFormatter(ABC, logging.Formatter):
    def __init__(self, message_prefix):
        """
        :param message_prefix:
            logging format string that will be prepended to message
        """
        super().__init__(f"{message_prefix}{{message}}", style="{")

    @abstractmethod
    def _format_message(self, state: RunnerState):
        pass

    def format(self, record: logging.LogRecord):
        # noinspection PyUnresolvedReferences
        state = record.state

        record.msg = self._format_message(state)

        return super().format(record)


class TxtMetricsFormatter(MetricsFormatter):
    """
    Translate batch metrics in human-readable format.

    This class is used by logging.Logger to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        super().__init__("[{asctime}] ")

    def _format_metrics(self, metrics):
        # metrics : dict[str: dict[str: float]]
        metrics_formatted = {}
        for key, value in metrics.items():
            metrics_formatted_ = [
                f"{m_name}={m_value:.4f}"
                for m_name, m_value in sorted(value.items())
            ]
            metrics_formatted_ = ' | '.join(metrics_formatted_)
            metrics_formatted[key] = metrics_formatted_

        return metrics_formatted

    def _format_message(self, state: RunnerState):
        message = [""]
        metrics = self._format_metrics(state.metrics.epoch_values)
        for key, value in metrics.items():
            message.append(
                f"{state.stage_epoch}/{state.num_epochs} "
                f"* Epoch {state.epoch} ({key}): {value}")
        message = "\n".join(message)
        return message


class JsonMetricsFormatter(MetricsFormatter):
    """
    Translate batch metrics in json format.

    This class is used by logging.Logger to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        super().__init__("")

    def _format_message(self, state: RunnerState):
        res = dict(
            metirics=state.metrics.epoch_values.copy(),
            epoch=state.epoch,
            time=datetime.now().isoformat()
        )
        return json.dumps(res, indent=True, ensure_ascii=False)


class ConsoleLogger(Callback):
    """
    Logger callback, translates state.metrics to console and text file
    """

    def __init__(self):
        self.logger = None

    @staticmethod
    def _get_logger(logdir):
        logger = logging.getLogger("metrics")
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(f"{logdir}/metrics.txt")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        # @TODO: fix json logger
        # jh = logging.FileHandler(f"{logdir}/metrics.json")
        # jh.setLevel(logging.INFO)

        txt_formatter = TxtMetricsFormatter()
        # json_formatter = JsonMetricsFormatter()
        fh.setFormatter(txt_formatter)
        ch.setFormatter(txt_formatter)
        # jh.setFormatter(json_formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        # logger.addHandler(jh)
        return logger

    def on_stage_start(self, state: RunnerState):
        assert state.logdir is not None
        state.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = self._get_logger(state.logdir)

    def on_stage_end(self, state):
        self.logger.handlers = []

    def on_epoch_end(self, state):
        self.logger.info("", extra={"state": state})


class TensorboardLogger(Callback):
    """
    Logger callback, translates state.metrics to tensorboard
    """

    def __init__(
        self,
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
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        assert self.log_on_batch_end or self.log_on_epoch_end, \
            "You have to log something!"

        self.loggers = dict()

    def on_loader_start(self, state):
        lm = state.loader_name
        if lm not in self.loggers:
            self.loggers[lm] = UtilsFactory.create_tflogger(
                logdir=state.logdir, name=lm
            )

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(list(metrics.keys()))
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_batch_end(self, state: RunnerState):
        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.metrics.batch_values
            self._log_metrics(
                metrics=metrics_,
                step=state.step,
                mode=mode,
                suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.metrics.epoch_values[mode]
            self._log_metrics(
                metrics=metrics_,
                step=state.epoch,
                mode=mode,
                suffix="/epoch"
            )
