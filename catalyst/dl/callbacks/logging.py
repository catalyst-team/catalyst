from typing import Dict, List  # isort:skip
import logging
import os
import sys
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from tqdm import tqdm

from catalyst.dl import utils
from catalyst.dl.core import LoggerCallback, RunnerState
from catalyst.dl.utils.formatters import TxtMetricsFormatter
from catalyst.utils import format_metric
from catalyst.utils.tensorboard import SummaryWriter


class VerboseLogger(LoggerCallback):
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
                if None default is ``["_timers/_fps"]``
                to remove always_show metrics set it to an empty list ``[]``
            never_show (List[str]): list of metrics which will not be shown
        """
        super().__init__()
        self.tqdm: tqdm = None
        self.step = 0
        self.always_show = (
            always_show if always_show is not None else ["_timers/_fps"]
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

    def on_loader_start(self, state: RunnerState):
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

    def on_batch_end(self, state: RunnerState):
        """Update tqdm progress bar at the end of each batch"""
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(state.metrics.batch_values.items())
                if self._need_show(k)
            }
        )
        self.tqdm.update()

    def on_loader_end(self, state: RunnerState):
        """Cleanup and close tqdm progress bar"""
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_exception(self, state: RunnerState):
        """Called if an Exception was raised"""
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
        """Init ``ConsoleLogger``"""
        super().__init__()
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

        txt_formatter = TxtMetricsFormatter()
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

    def on_stage_start(self, state: RunnerState):
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
        Translate ``state.metrics`` to console and text file
        at the end of an epoch
        """
        self.logger.info("", extra={"state": state})


class TensorboardLogger(LoggerCallback):
    """
    Logger callback, translates ``state.metrics`` to tensorboard
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
        super().__init__()
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

    def on_batch_end(self, state: RunnerState):
        """Translate batch metrics to tensorboard"""
        if state.logdir is None:
            return

        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.metrics.batch_values
            self._log_metrics(
                metrics=metrics_, step=state.step, mode=mode, suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        """Translate epoch metrics to tensorboard"""
        if state.logdir is None:
            return

        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.metrics.epoch_values[mode]
            self._log_metrics(
                metrics=metrics_,
                step=state.epoch_log,
                mode=mode,
                suffix="/epoch",
            )
        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, state: RunnerState):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


class TelegramLogger(LoggerCallback):
    """
    Logger callback, translates ``state.metrics`` to telegram channel
    """

    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        metric_names: List[str] = None,
        log_on_stage_start: bool = True,
        log_on_loader_start: bool = True,
        log_on_loader_end: bool = True,
        log_on_stage_end: bool = True,
        log_on_exception: bool = True,
    ):
        """
        Args:
            token (str): telegram bot's token,
                see https://core.telegram.org/bots
            chat_id (str): Chat unique identifier
            metric_names: List of metric names to log.
                if none - logs everything.
            log_on_stage_start (bool): send notification on stage start
            log_on_loader_start (bool): send notification on loader start
            log_on_loader_end (bool): send notification on loader end
            log_on_stage_end (bool): send notification on stage end
            log_on_exception (bool): send notification on exception
        """
        super().__init__()
        # @TODO: replace this logic with global catalyst config at ~/.catalyst
        self._token = token or os.environ.get("CATALYST_TELEGRAM_TOKEN", None)
        self._chat_id = (
            chat_id or os.environ.get("CATALYST_TELEGRAM_CHAT_ID", None)
        )
        assert self._token is not None and self._chat_id is not None
        self._base_url = (
            f"https://api.telegram.org/bot{self._token}/sendMessage"
        )

        self.log_on_stage_start = log_on_stage_start
        self.log_on_loader_start = log_on_loader_start
        self.log_on_loader_end = log_on_loader_end
        self.log_on_stage_end = log_on_stage_end
        self.log_on_exception = log_on_exception

        self.metrics_to_log = metric_names

    def _send_text(self, text: str):
        try:
            url = (
                f"{self._base_url}?"
                f"chat_id={self._chat_id}&"
                f"disable_web_page_preview=1&"
                f"text={quote_plus(text, safe='')}"
            )

            request = Request(url)
            urlopen(request)
        except Exception as e:
            logging.getLogger(__name__).warning(f"telegram.send.error:{e}")

    def on_stage_start(self, state: RunnerState):
        """Notify about starting a new stage"""
        if self.log_on_stage_start:
            text = f"{state.stage} stage was started"

            self._send_text(text)

    def on_loader_start(self, state: RunnerState):
        """Notify about starting running the new loader"""
        if self.log_on_loader_start:
            text = f"{state.loader_name} {state.epoch} epoch was started"

            self._send_text(text)

    def on_loader_end(self, state: RunnerState):
        """Translate ``state.metrics`` to telegram channel"""
        if self.log_on_loader_end:
            metrics = state.metrics.epoch_values[state.loader_name]

            if self.metrics_to_log is None:
                metrics_to_log = sorted(list(metrics.keys()))
            else:
                metrics_to_log = self.metrics_to_log

            rows: List[str] = [
                f"{state.loader_name} {state.epoch} epoch was finished:"
            ]

            for name in metrics_to_log:
                if name in metrics:
                    rows.append(format_metric(name, metrics[name]))

            text = "\n".join(rows)

            self._send_text(text)

    def on_stage_end(self, state: RunnerState):
        """Notify about finishing a stage"""
        if self.log_on_stage_end:
            text = f"{state.stage} stage was finished"

            self._send_text(text)

    def on_exception(self, state: RunnerState):
        """Notify about raised Exception"""
        if self.log_on_exception:
            exception = state.exception
            if utils.is_exception(exception) and not isinstance(
                exception, KeyboardInterrupt
            ):
                text = (
                    f"`{type(exception).__name__}` exception was raised:\n"
                    f"{exception}"
                )

                self._send_text(text)


__all__ = [
    "ConsoleLogger",
    "TelegramLogger",
    "TensorboardLogger",
    "VerboseLogger",

]
