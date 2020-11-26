from typing import Dict, List, TYPE_CHECKING
import logging
import os
import sys

from tqdm import tqdm

from catalyst.callbacks.formatters import TxtMetricsFormatter
from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.utils.misc import is_exception, split_dict_to_subdicts

if TYPE_CHECKING:
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
            always_show: list of metrics to always show
                if None default is ``["_timer/_fps"]``
                to remove always_show metrics set it to an empty list ``[]``
            never_show: list of metrics which will not be shown
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

    def on_loader_start(self, runner: "IRunner"):
        """Init tqdm progress bar."""
        self.step = 0
        self.tqdm = tqdm(
            total=runner.loader_len,
            desc=f"{runner.epoch}/{runner.num_epochs}"
            f" * Epoch ({runner.loader_key})",
            leave=True,
            ncols=0,
            file=sys.stdout,
        )

    def on_batch_end(self, runner: "IRunner"):
        """Update tqdm progress bar at the end of each batch."""
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(runner.batch_metrics.items())
                if self._need_show(k)
            }
        )
        self.tqdm.update()

    def on_loader_end(self, runner: "IRunner"):
        """Cleanup and close tqdm progress bar."""
        # self.tqdm.visible = False
        # self.tqdm.leave = True
        # self.tqdm.disable = True
        self.tqdm.clear()
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_exception(self, runner: "IRunner"):
        """Called if an Exception was raised."""
        exception = runner.exception
        if not is_exception(exception):
            return

        if isinstance(exception, KeyboardInterrupt):
            if self.tqdm is not None:
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
    def _get_logger():
        logger = logging.getLogger("metrics_logger")
        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def _setup_logger(logger, logdir: str):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        txt_formatter = TxtMetricsFormatter()
        ch.setFormatter(txt_formatter)

        # add the handlers to the logger
        logger.addHandler(ch)

        if logdir:
            fh = logging.FileHandler(f"{logdir}/log.txt")
            fh.setLevel(logging.INFO)
            fh.setFormatter(txt_formatter)
            logger.addHandler(fh)
        # logger.addHandler(jh)

    @staticmethod
    def _clean_logger(logger):
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

    def on_stage_start(self, runner: "IRunner"):
        """Prepare ``runner.logdir`` for the current stage."""
        if runner.logdir:
            runner.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = self._get_logger()
        self._clean_logger(self.logger)
        self._setup_logger(self.logger, runner.logdir)

    def on_epoch_end(self, runner: "IRunner"):
        """
        Translate ``runner.metric_manager`` to console and text file
        at the end of an epoch.

        Args:
            runner: current runner instance
        """
        self.logger.info("", extra={"runner": runner})

    def on_stage_end(self, runner: "IRunner"):
        """Called at the end of each stage."""
        self._clean_logger(self.logger)


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
            metric_names: list of metric names to log,
                if none - logs everything
            log_on_batch_end: logs per-batch metrics if set True
            log_on_epoch_end: logs per-epoch metrics if set True
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

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook. Check ``logdir`` correctness.

        Args:
            runner: current runner
        """
        assert runner.logdir is not None

        extra_mode = "_base"
        log_dir = os.path.join(runner.logdir, f"{extra_mode}_log")
        self.loggers[extra_mode] = SummaryWriter(log_dir)

    def on_loader_start(self, runner: "IRunner"):
        """Prepare tensorboard writers for the current stage."""
        if runner.loader_key not in self.loggers:
            log_dir = os.path.join(runner.logdir, f"{runner.loader_key}_log")
            self.loggers[runner.loader_key] = SummaryWriter(log_dir)

    def on_batch_end(self, runner: "IRunner"):
        """Translate batch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_batch_end:
            mode = runner.loader_key
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix="/batch",
            )

    def on_epoch_end(self, runner: "IRunner"):
        """Translate epoch metrics to tensorboard."""
        if runner.logdir is None:
            return

        if self.log_on_epoch_end:
            per_mode_metrics = split_dict_to_subdicts(
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

    def on_stage_end(self, runner: "IRunner"):
        """Close opened tensorboard writers."""
        if runner.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


class CSVLogger(ILoggerCallback):
    """Logs metrics to csv file on epoch end"""

    def __init__(
        self, metric_names: List[str] = None,
    ):
        """
        Args:
            metric_names: list of metric names to log,
                if none - logs everything
        """
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.metrics_to_log = metric_names

        self.loggers = {}
        self.header_created = {}

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        On loader start action.

        Args:
            runner: current runner
        """
        if runner.loader_key not in self.loggers:
            log_dir = os.path.join(runner.logdir, f"{runner.loader_key}_log")
            os.makedirs(log_dir, exist_ok=True)
            self.loggers[runner.loader_key] = open(
                os.path.join(log_dir, "logs.csv"), "a+"
            )
            self.header_created[runner.loader_key] = False

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, loader_key: str
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        log_line_csv = f"{step},"
        for metric in metrics_to_log:
            log_line_csv += str(metrics[metric]) + ","
        log_line_csv = (
            log_line_csv[:-1] + "\n"
        )  # replace last "," with new line
        self.loggers[loader_key].write(log_line_csv)

    def _make_header(self, metrics: Dict[str, float], loader_key: str):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log
        log_line_header = "step,"
        for metric in metrics_to_log:
            log_line_header += metric + ","
        log_line_header = (
            log_line_header[:-1] + "\n"
        )  # replace last "," with new line
        self.loggers[loader_key].write(log_line_header)

    def on_epoch_end(self, runner: "IRunner"):
        """
        Logs metrics here

        Args:
            runner: runner for experiment
        """
        if runner.logdir is None:
            return
        per_loader_metrics = split_dict_to_subdicts(
            dct=runner.epoch_metrics,
            prefixes=list(runner.loaders.keys()),
            extra_key="_base",
        )
        for loader_key, per_loader_metrics in per_loader_metrics.items():
            if "base" in loader_key:
                continue
            if not self.header_created[loader_key]:
                self._make_header(
                    metrics=per_loader_metrics, loader_key=loader_key
                )
                self.header_created[loader_key] = True
            self._log_metrics(
                metrics=per_loader_metrics,
                step=runner.global_epoch,
                loader_key=loader_key,
            )

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        Closes loggers

        Args:
            runner: runner for experiment
        """
        for _k, logger in self.loggers.items():
            logger.close()


__all__ = [
    "ILoggerCallback",
    "ConsoleLogger",
    "TensorboardLogger",
    "VerboseLogger",
    "CSVLogger",
]
