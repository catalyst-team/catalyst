from typing import List
import logging
import os
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from catalyst import utils
from catalyst.core import Callback, CallbackNode, CallbackOrder, State


class TelegramLogger(Callback):
    """
    Logger callback, translates ``state.metric_manager`` to telegram channel.
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
        super().__init__(order=CallbackOrder.Logging, node=CallbackNode.Master)
        # @TODO: replace this logic with global catalyst config at ~/.catalyst
        self._token = token or os.environ.get("CATALYST_TELEGRAM_TOKEN", None)
        self._chat_id = chat_id or os.environ.get(
            "CATALYST_TELEGRAM_CHAT_ID", None
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

    def on_stage_start(self, state: State):
        """Notify about starting a new stage."""
        if self.log_on_stage_start:
            text = f"{state.stage_name} stage was started"

            self._send_text(text)

    def on_loader_start(self, state: State):
        """Notify about starting running the new loader."""
        if self.log_on_loader_start:
            text = (
                f"{state.loader_name} {state.global_epoch} epoch has started"
            )

            self._send_text(text)

    def on_loader_end(self, state: State):
        """Translate ``state.metric_manager`` to telegram channel."""
        if self.log_on_loader_end:
            metrics = state.loader_metrics

            if self.metrics_to_log is None:
                metrics_to_log = sorted(metrics.keys())
            else:
                metrics_to_log = self.metrics_to_log

            rows: List[str] = [
                f"{state.loader_name} {state.global_epoch} epoch was finished:"
            ]

            for name in metrics_to_log:
                if name in metrics:
                    rows.append(utils.format_metric(name, metrics[name]))

            text = "\n".join(rows)

            self._send_text(text)

    def on_stage_end(self, state: State):
        """Notify about finishing a stage."""
        if self.log_on_stage_end:
            text = f"{state.stage_name} stage was finished"

            self._send_text(text)

    def on_exception(self, state: State):
        """Notify about raised ``Exception``."""
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
