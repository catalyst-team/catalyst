# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List
import logging
import os
import sys

from tqdm import tqdm

from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.callbacks import formatters
from catalyst.core.runner import IRunner


class ILoggerCallback(Callback):
    """Logger callback interface, abstraction over logging step"""

    pass


class VerboseLogger(ILoggerCallback):
    """Logs the params into console."""

    def __init__(
        self, always_show: List[str] = None, never_show: List[str] = None,
    ):
        """
        Args:
            always_show (List[str]): list of metrics to always show
                if None default is ``["_timer/_fps"]``
                to remove always_show metrics set it to an empty list ``[]``
            never_show (List[str]): list of metrics which will not be shown
        """
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.tqdm: tqdm = None
        self.step = 0
        self.always_show = (
            always_show if always_show is not None else ["_timer/_fps"]
        )
        self.never_show = never_show if never_show is not None else []

        intersection = set(self.always_show) & set(self.never_show)

        error_message = (
            f"Intersection of always_show and "
            f"never_show has common values: {intersection}"
        )
        if bool(intersection):
            raise ValueError(error_message)

    def _need_show(self, key: str):
        not_is_never_shown: bool = key not in self.never_show
        is_always_shown: bool = key in self.always_show
        not_basic = not (key.startswith("_base") or key.startswith("_timer"))

        result = not_is_never_shown and (is_always_shown or not_basic)

        return result

    def on_loader_start(self, runner: IRunner):
        """Init tqdm progress bar."""
        self.step = 0
        self.tqdm = tqdm(
            total=runner.loader_len,
            desc=f"{runner.epoch}/{runner.num_epochs}"
            f" * Epoch ({runner.loader_name})",
            leave=True,
            ncols=0,
            file=sys.stdout,
        )

    def on_loader_end(self, runner: IRunner):
        """Cleanup and close tqdm progress bar."""
        # self.tqdm.visible = False
        # self.tqdm.leave = True
        # self.tqdm.disable = True
        self.tqdm.clear()
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_batch_end(self, runner: IRunner):
        """Update tqdm progress bar at the end of each batch."""
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(runner.batch_metrics.items())
                if self._need_show(k)
            }
        )
        self.tqdm.update()

    def on_exception(self, runner: IRunner):
        """Called if an Exception was raised."""
        exception = runner.exception
        if not utils.is_exception(exception):
            return

        if isinstance(exception, KeyboardInterrupt):
            self.tqdm.write("Early exiting")
            runner.need_exception_reraise = False


class ConsoleLogger(ILoggerCallback):
    """Logger callback,
    translates ``runner.*_metrics`` to console and text file.
    """

    def __init__(self):
        """Init ``ConsoleLogger``."""
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.logger = None

    @staticmethod
    def _get_logger(logdir):
        logger = logging.getLogger("metrics_logger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        txt_formatter = formatters.TxtMetricsFormatter()
        ch.setFormatter(txt_formatter)

        # add the handlers to the logger
        logger.addHandler(ch)

        if logdir:
            fh = logging.FileHandler(f"{logdir}/log.txt")
            fh.setLevel(logging.INFO)
            fh.setFormatter(txt_formatter)
            logger.addHandler(fh)

        # logger.addHandler(jh)
        return logger

    def on_stage_start(self, runner: IRunner):
        """Prepare ``runner.logdir`` for the current stage."""
        if runner.logdir:
            runner.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = self._get_logger(runner.logdir)

    def on_stage_end(self, runner: IRunner):
        """Called at the end of each stage."""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers = []

    def on_epoch_end(self, runner: IRunner):
        """
        Translate ``runner.metric_manager`` to console and text file
        at the end of an epoch.

        Args:
            runner (IRunner): current runner instance
        """
        self.logger.info("", extra={"runner": runner})


class TensorboardLogger(ILoggerCallback):
    """Logger callback, translates ``runner.metric_manager`` to tensorboard."""

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
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        self.loggers = {}

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_stage_start(self, runner: IRunner):
        """@TODO: Docs. Contribution is welcome."""
        assert runner.logdir is not None

        extra_mode = "_base"
        log_dir = os.path.join(runner.logdir, f"{extra_mode}_log")
        self.loggers[extra_mode] = SummaryWriter(log_dir)

    def on_loader_start(self, runner: IRunner):
        """Prepare tensorboard writers for the current stage."""
        if runner.loader_name not in self.loggers:
            log_dir = os.path.join(runner.logdir, f"{runner.loader_name}_log")
            self.loggers[runner.loader_name] = SummaryWriter(log_dir)

    def on_batch_end(self, runner: IRunner):
        """Translate batch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_batch_end:
            mode = runner.loader_name
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix="/batch",
            )

    def on_epoch_end(self, runner: IRunner):
        """Translate epoch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_epoch_end:
            per_mode_metrics = utils.split_dict_to_subdicts(
                dct=runner.epoch_metrics,
                prefixes=list(runner.loaders.keys()),
                extra_key="_base",
            )

            for mode, metrics in per_mode_metrics.items():
                # suffix = "" if mode == "_base" else "/epoch"
                self._log_metrics(
                    metrics=metrics,
                    step=runner.global_epoch,
                    mode=mode,
                    suffix="/epoch",
                )

        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, runner: IRunner):
        """Close opened tensorboard writers."""
        if runner.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


__all__ = [
    "ILoggerCallback",
    "ConsoleLogger",
    "TensorboardLogger",
    "VerboseLogger",
]
