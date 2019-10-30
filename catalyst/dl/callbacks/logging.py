from typing import List, Dict, Optional
import os
import sys
import logging
from urllib.request import Request, urlopen
from urllib.parse import quote_plus
from tqdm import tqdm

from catalyst.dl.core import LoggerCallback, RunnerState
from catalyst.dl.utils.formatters import TxtMetricsFormatter
from catalyst.dl import utils
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


class TelegramLogger(LoggerCallback):
    """
    Logger callback, translates state.metrics to telegram channel
    """

    def __init__(
            self,
            token: str,
            chat_id: str,
            log_on_stage_start: bool = True,
            log_on_loader_start: bool = True,
            log_on_loader_end: bool = True,
            log_on_stage_end: bool = True
    ):
        """

        :param token: Telegram bot's token, see https://core.telegram.org/bots
        :param chat_id: Chat unique identifier:
            1. Add your bot to a group.
            2. Send a dummy message like this: /test @your_bot_name_here
            3. Open this page: https://api.telegram.org/bot<token>/getUpdates,
               where <token> is yours bot token.
               example: https://api.telegram.org/bot1234566789/getUpdates,
               where token = 1234566789.
            4. You should look for this pattern ..."chat":{"id":<chat_id>...,
               where <chat_id> - id of your group.
        :param log_on_stage_start: Send notification on stage start.
        :param log_on_loader_start: Send notification on loader start.
        :param log_on_loader_end: Send notification on loader end.
        :param log_on_stage_end: Send notification on stage end.
        """
        super().__init__()
        self._token = token
        self._chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{self._token}" \
                         f"/sendMessage"

        self._log_on_stage_start = log_on_stage_start
        self._log_on_loader_start = log_on_loader_start
        self._log_on_loader_end = log_on_loader_end
        self._log_on_stage_end = log_on_stage_end

        self._metrics_to_log: Optional[List[str]] = None

    def _send_text(self, text: str):
        try:
            url = f"{self._base_url}?" \
                  f"chat_id={self._chat_id}&" \
                  f"disable_web_page_preview=1&" \
                  f"text={quote_plus(text, safe='')}"

            request = Request(url)
            urlopen(request)
        except Exception as e:
            print(f"telegram.send.error:{e}")

    def on_stage_start(self, state: RunnerState):
        if self._log_on_stage_start:
            text = f"{state.stage} stage was started"

            self._send_text(text)

    def on_loader_start(self, state: RunnerState):
        if self._log_on_loader_start:
            text = f"{state.loader_name} {state.epoch} epoch was started"

            self._send_text(text)

    def on_loader_end(self, state: RunnerState):
        if self._log_on_loader_end:
            metrics = state.metrics.epoch_values[state.loader_name]

            if self._metrics_to_log is None:
                self._metrics_to_log = sorted(list(metrics.keys()))

            rows: List[str] = [
                f"{state.loader_name} {state.epoch} epoch was finished:",
            ]

            for name in self._metrics_to_log:
                if name in metrics:
                    rows.append(f"{name}={metrics[name]}")

            text = "\n".join(rows)

            self._send_text(text)

    def on_stage_end(self, state: RunnerState):
        if self._log_on_stage_end:
            text = f"{state.stage} stage was finished"

            self._send_text(text)


__all__ = [
    "VerboseLogger",
    "ConsoleLogger",
    "TensorboardLogger",
    "TelegramLogger"
]
