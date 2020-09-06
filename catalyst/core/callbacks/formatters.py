# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict
from abc import ABC, abstractmethod
import logging

from catalyst.core import utils
from catalyst.core.runner import IRunner


class MetricsFormatter(ABC, logging.Formatter):
    """Abstract metrics formatter."""

    def __init__(self, message_prefix):
        """
        Args:
            message_prefix: logging format string
                that will be prepended to message
        """
        super().__init__(f"{message_prefix}{{message}}", style="{")

    @abstractmethod
    def _format_message(self, runner: IRunner):
        pass

    def format(self, record: logging.LogRecord):
        """Format message string."""
        # noinspection PyUnresolvedReferences
        runner = record.runner

        record.msg = self._format_message(runner)

        return super().format(record)


class TxtMetricsFormatter(MetricsFormatter):
    """Translate batch metrics in human-readable format.

    This class is used by ``logging.Logger`` to make a string from record.
    For details refer to official docs for 'logging' module.

    .. warning::
        This is inner class used by Logger callback,
        no need to use it directly!
    """

    def __init__(self):
        """
        Initializes the ``TxtMetricsFormatter``
        """
        super().__init__("[{asctime}] ")

    def _format_metrics(self, metrics: Dict[str, Dict[str, float]]):
        metrics_formatted = {}
        for key, value in metrics.items():
            metrics_formatted_part = [
                utils.format_metric(m_name, m_value)
                for m_name, m_value in sorted(value.items())
            ]
            metrics_formatted_part = " | ".join(metrics_formatted_part)
            metrics_formatted[key] = metrics_formatted_part

        return metrics_formatted

    def _format_message(self, runner: IRunner):
        message = [""]
        mode_metrics = utils.split_dict_to_subdicts(
            dct=runner.epoch_metrics,
            prefixes=list(runner.loaders.keys()),
            extra_key="_base",
        )
        metrics = self._format_metrics(mode_metrics)
        for key, value in metrics.items():
            message.append(
                f"{runner.epoch}/{runner.num_epochs} "
                f"* Epoch {runner.global_epoch} ({key}): {value}"
            )
        message = "\n".join(message)
        return message


__all__ = ["MetricsFormatter", "TxtMetricsFormatter"]
