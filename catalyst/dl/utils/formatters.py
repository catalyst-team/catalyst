from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging

from catalyst.dl.core.state import RunnerState
from catalyst.utils import format_metric


class MetricsFormatter(ABC, logging.Formatter):
    """
    Abstract metrics formatter
    """

    def __init__(self, message_prefix):
        """
        Args:
            message_prefix: logging format string
                that will be prepended to message
        """
        super().__init__(f"{message_prefix}{{message}}", style="{")

    @abstractmethod
    def _format_message(self, state: RunnerState):
        pass

    def format(self, record: logging.LogRecord):
        """
        Format message string
        """
        # noinspection PyUnresolvedReferences
        state = record.state

        record.msg = self._format_message(state)

        return super().format(record)


class TxtMetricsFormatter(MetricsFormatter):
    """
    Translate batch metrics in human-readable format.

    This class is used by ``logging.Logger`` to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        """
        Initializes the ``TxtMetricsFormatter``
        """
        super().__init__("[{asctime}] ")

    def _format_metrics(self, metrics):
        # metrics : dict[str: dict[str: float]]
        metrics_formatted = {}
        for key, value in metrics.items():
            metrics_formatted_ = [
                format_metric(m_name, m_value)
                for m_name, m_value in sorted(value.items())
            ]
            metrics_formatted_ = " | ".join(metrics_formatted_)
            metrics_formatted[key] = metrics_formatted_

        return metrics_formatted

    def _format_message(self, state: RunnerState):
        message = [""]
        metrics = self._format_metrics(state.metrics.epoch_values)
        for key, value in metrics.items():
            message.append(
                f"{state.stage_epoch_log}/{state.num_epochs} "
                f"* Epoch {state.epoch_log} ({key}): {value}"
            )
        message = "\n".join(message)
        return message


class JsonMetricsFormatter(MetricsFormatter):
    """
    Translate batch metrics in json format.

    This class is used by ``logging.Logger`` to make a string from record.
    For details refer to official docs for 'logging' module.

    Note:
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        """
        Initializes the ``JsonMetricsFormatter``
        """
        super().__init__("")

    def _format_message(self, state: RunnerState):
        res = dict(
            metirics=state.metrics.epoch_values.copy(),
            epoch=state.epoch,
            time=datetime.now().isoformat()
        )
        return json.dumps(res, indent=True, ensure_ascii=False)


__all__ = ["MetricsFormatter", "TxtMetricsFormatter", "JsonMetricsFormatter"]
