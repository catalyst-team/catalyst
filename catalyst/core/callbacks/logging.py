from typing import Any, Dict, List  # isort:skip
from collections import defaultdict
import logging
import os
import sys

from tqdm import tqdm

from catalyst import utils
from catalyst.core import _State, Callback, CallbackNode, CallbackOrder
from catalyst.utils import meters
from catalyst.utils.tools.tensorboard import SummaryWriter
from . import formatters


class MetricsManagerCallback(Callback):
    """
    Prepares metrics for logging, transferring values from PyTorch to numpy
    """
    def __init__(self):
        super().__init__(
            order=CallbackOrder.Logging - 1,
            node=CallbackNode.All,
        )
        self.meters: Dict[str, meters.AverageValueMeter] = None

    @staticmethod
    def _to_single_value(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()

        value = float(value)
        return value

    @staticmethod
    def _process_metrics(metrics: Dict[str, Any]):
        output = {}
        for key, value in metrics.items():
            value = MetricsManagerCallback._to_single_value(value)
            value = utils.distributed_mean(value)
            output[key] = value
        return output

    def on_epoch_start(self, state: _State):
        state.epoch_metrics = defaultdict(None)

    def on_loader_start(self, state: _State):
        state.loader_metrics = defaultdict(None)
        self.meters = defaultdict(meters.AverageValueMeter)

    def on_loader_end(self, state: _State):
        for key, value in self.meters.items():
            value = value.mean
            state.loader_metrics[key] = value
            state.epoch_metrics[f"{state.loader_name}_{key}"] = value

    def on_batch_start(self, state: _State):
        state.batch_metrics = defaultdict(None)

    def on_batch_end(self, state: _State):
        state.batch_metrics = self._process_metrics(state.batch_metrics)
        for key, value in state.batch_metrics.items():
            self.meters[key].add(value)


class VerboseLogger(Callback):
    """
    Logs the params into console
    """
    def __init__(
        self,
        always_show: List[str] = None,
        never_show: List[str] = None,
    ):
        """
        Args:
            always_show (List[str]): list of metrics to always show
                if None default is ``["_timer/_fps"]``
                to remove always_show metrics set it to an empty list ``[]``
            never_show (List[str]): list of metrics which will not be shown
        """
        super().__init__(order=CallbackOrder.Logging, node=CallbackNode.Master)
        self.tqdm: tqdm = None
        self.step = 0
        self.always_show = (
            always_show if always_show is not None else ["_timer/_fps"]
        )
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

    def on_loader_start(self, state: _State):
        """Init tqdm progress bar"""
        self.step = 0
        self.tqdm = tqdm(
            total=state.loader_len,
            desc=f"{state.stage_epoch_log}/{state.num_epochs}"
            f" * Epoch ({state.loader_name})",
            leave=True,
            ncols=0,
            file=sys.stdout,
        )

    def on_loader_end(self, state: _State):
        """Cleanup and close tqdm progress bar"""
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_batch_end(self, state: _State):
        """Update tqdm progress bar at the end of each batch"""
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(state.batch_metrics.items())
                if self._need_show(k)
            }
        )
        self.tqdm.update()

    def on_exception(self, state: _State):
        """Called if an Exception was raised"""
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if isinstance(exception, KeyboardInterrupt):
            self.tqdm.write("Early exiting")
            state.need_exception_reraise = False


class ConsoleLogger(Callback):
    """
    Logger callback,
    translates ``state.*_metrics`` to console and text file
    """
    def __init__(self):
        """Init ``ConsoleLogger``"""
        super().__init__(order=CallbackOrder.Logging, node=CallbackNode.Master)
        self.logger = None

    @staticmethod
    def _get_logger(logdir):
        logger = logging.getLogger("metrics_logger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        # @TODO: fix json logger
        # jh = logging.FileHandler(f"{logdir}/metrics.json")
        # jh.setLevel(logging.INFO)

        txt_formatter = formatters.TxtMetricsFormatter()
        # json_formatter = JsonMetricsFormatter()
        ch.setFormatter(txt_formatter)
        # jh.setFormatter(json_formatter)

        # add the handlers to the logger
        logger.addHandler(ch)

        if logdir:
            fh = logging.FileHandler(f"{logdir}/log.txt")
            fh.setLevel(logging.INFO)
            fh.setFormatter(txt_formatter)
            logger.addHandler(fh)

        # logger.addHandler(jh)
        return logger

    def on_stage_start(self, state: _State):
        """Prepare ``state.logdir`` for the current stage"""
        if state.logdir:
            state.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = self._get_logger(state.logdir)

    def on_stage_end(self, state):
        """Called at the end of each stage"""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers = []

    def on_epoch_end(self, state):
        """
        Translate ``state.metric_manager`` to console and text file
        at the end of an epoch
        """
        self.logger.info("", extra={"state": state})


class TensorboardLogger(Callback):
    """
    Logger callback, translates ``state.metric_manager`` to tensorboard
    """
    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
    ):
        """
        Args:
            metric_names (List[str]): list of metric names to log,
                if none - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
        """
        super().__init__(order=CallbackOrder.Logging, node=CallbackNode.Master)
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

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
        """Prepare tensorboard writers for the current stage"""
        if state.logdir is None:
            return

        lm = state.loader_name
        if lm not in self.loggers:
            log_dir = os.path.join(state.logdir, f"{lm}_log")
            self.loggers[lm] = SummaryWriter(log_dir)

    def on_batch_end(self, state: _State):
        """Translate batch metrics to tensorboard"""
        if state.logdir is None:
            return

        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.batch_metrics
            self._log_metrics(
                metrics=metrics_, step=state.step, mode=mode, suffix="/batch"
            )

    def on_loader_end(self, state: _State):
        """Translate epoch metrics to tensorboard"""
        if state.logdir is None:
            return

        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.loader_metrics
            self._log_metrics(
                metrics=metrics_,
                step=state.epoch_log,
                mode=mode,
                suffix="/epoch",
            )
        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, state: _State):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


__all__ = [
    "ConsoleLogger",
    "TensorboardLogger",
    "VerboseLogger",
    "MetricsManagerCallback",
]
