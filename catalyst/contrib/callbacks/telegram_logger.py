# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List, TYPE_CHECKING
import logging
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.settings import SETTINGS
from catalyst.utils.misc import format_metric, is_exception

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

logger = logging.getLogger(__name__)


class TelegramLogger(Callback):
    """
    Logger callback, translates ``runner.metric_manager`` to telegram channel.
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
            token: telegram bot's token,
                see https://core.telegram.org/bots
            chat_id: Chat unique identifier
            metric_names: List of metric names to log.
                if none - logs everything.
            log_on_stage_start: send notification on stage start
            log_on_loader_start: send notification on loader start
            log_on_loader_end: send notification on loader end
            log_on_stage_end: send notification on stage end
            log_on_exception: send notification on exception
        """
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        # @TODO: replace this logic with global catalyst config at ~/.catalyst
        self._token = token or SETTINGS.telegram_logger_token
        self._chat_id = chat_id or SETTINGS.telegram_logger_chat_id
        assert self._token is not None and self._chat_id is not None
        self._base_url = f"https://api.telegram.org/bot{self._token}/sendMessage"

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
            urlopen(request)  # noqa: S310
        except Exception as e:
            logger.warning(f"telegram.send.error:{e}")

    def on_stage_start(self, runner: "IRunner"):
        """Notify about starting a new stage."""
        if self.log_on_stage_start:
            text = f"{runner.stage} stage was started"

            self._send_text(text)

    def on_loader_start(self, runner: "IRunner"):
        """Notify about starting running the new loader."""
        if self.log_on_loader_start:
            text = f"{runner.loader_key} {runner.global_epoch} epoch has started"

            self._send_text(text)

    def on_loader_end(self, runner: "IRunner"):
        """Translate ``runner.metric_manager`` to telegram channel."""
        if self.log_on_loader_end:
            metrics = runner.loader_metrics

            if self.metrics_to_log is None:
                metrics_to_log = sorted(metrics.keys())
            else:
                metrics_to_log = self.metrics_to_log

            rows: List[str] = [
                f"{runner.loader_key} {runner.global_epoch}" f" epoch was finished:"
            ]

            for name in metrics_to_log:
                if name in metrics:
                    rows.append(format_metric(name, metrics[name]))

            text = "\n".join(rows)

            self._send_text(text)

    def on_stage_end(self, runner: "IRunner"):
        """Notify about finishing a stage."""
        if self.log_on_stage_end:
            text = f"{runner.stage} stage was finished"

            self._send_text(text)

    def on_exception(self, runner: "IRunner"):
        """Notify about raised ``Exception``."""
        if self.log_on_exception:
            exception = runner.exception
            if is_exception(exception) and not isinstance(exception, KeyboardInterrupt):
                text = f"`{type(exception).__name__}` exception was raised:\n" f"{exception}"

                self._send_text(text)


__all__ = ["TelegramLogger"]
