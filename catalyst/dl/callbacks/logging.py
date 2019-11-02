from typing import Dict, List  # isort:skip
import logging
import os
import sys

from tqdm import tqdm

from catalyst.dl import utils
from catalyst.dl.core import LoggerCallback, RunnerState
from catalyst.dl.utils.formatters import TxtMetricsFormatter
from catalyst.utils.tensorboard import SummaryWriter


class VerboseLogger(LoggerCallback):
    def __init__(
        self,
        always_show: List[str] = None,
        never_show: List[str] = None
    ):
        """
        Logs the params into console

        Args:
            always_show (List[str]): list of metrics to always show
                if None default is ``["_timers/_fps"]``
                to remove always_show metrics set it to an empty list ``[]``
            never_show (List[str]): list of metrics which will not be shown
        """
        super().__init__()
        self.tqdm: tqdm = None
        self.step = 0
        self.always_show = always_show \
            if always_show is not None else ["_timers/_fps"]
        self.never_show = never_show if never_show is not None else []

        intersection = set(self.always_show) & set(self.never_show)

        _error_message = (
            f"Intersection of always_show and "
            f"never_show has common values: {intersection}"
        )
        if bool(intersection):
            raise ValueError(_error_message)

    def _need_show(self, key: str):
        not_is_never_shown: bool = key not in self.never_show
        is_always_shown: bool = key in self.always_show
        not_basic = not (key.startswith("_base") or key.startswith("_timers"))

        result = not_is_never_shown and (is_always_shown or not_basic)

        return result

    def on_loader_start(self, state: RunnerState):
        self.step = 0
        self.tqdm = tqdm(
            total=state.loader_len,
            desc=f"{state.stage_epoch_log}/{state.num_epochs}"
            f" * Epoch ({state.loader_name})",
            leave=True,
            ncols=0,
            file=sys.stdout
        )

    def on_batch_end(self, state: RunnerState):
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(state.metrics.batch_values.items())
                if self._need_show(k)
            }
        )
        self.tqdm.update()

    def on_loader_end(self, state: RunnerState):
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_exception(self, state: RunnerState):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if isinstance(exception, KeyboardInterrupt):
            self.tqdm.write("Early exiting")
            state.need_reraise_exception = False


class ConsoleLogger(LoggerCallback):
    """
    Logger callback, translates ``state.metrics`` to console and text file
    """
    def __init__(self):
        super().__init__()
        self.logger = None

    @staticmethod
    def _get_logger(logdir):
        logger = logging.getLogger("metrics_logger")
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(f"{logdir}/log.txt")
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
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers = []

    def on_epoch_end(self, state):
        self.logger.info("", extra={"state": state})


class TensorboardLogger(LoggerCallback):
    """
    Logger callback, translates state.metrics to tensorboard
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True
    ):
        """
        Args:
            metric_names: List of metric names to log.
                If none - logs everything.
            log_on_batch_end: Logs per-batch metrics if set True.
            log_on_epoch_end: Logs per-epoch metrics if set True.
        """
        super().__init__()
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        assert self.log_on_batch_end or self.log_on_epoch_end, \
            "You have to log something!"

        self.loggers = dict()

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

    def on_loader_start(self, state):
        lm = state.loader_name
        if lm not in self.loggers:
            log_dir = os.path.join(state.logdir, f"{lm}_log")
            self.loggers[lm] = SummaryWriter(log_dir)

    def on_batch_end(self, state: RunnerState):
        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.metrics.batch_values
            self._log_metrics(
                metrics=metrics_, step=state.step, mode=mode, suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.metrics.epoch_values[mode]
            self._log_metrics(
                metrics=metrics_,
                step=state.epoch_log,
                mode=mode,
                suffix="/epoch"
            )
        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, state: RunnerState):
        for logger in self.loggers.values():
            logger.close()


__all__ = [
    "VerboseLogger",
    "ConsoleLogger",
    "TensorboardLogger"
]
