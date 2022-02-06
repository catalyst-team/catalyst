from typing import Any, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ILogger:
    """An abstraction that syncs experiment run with monitoring tools.

    Args:
        log_batch_metrics: boolean flag to log batch metrics.
        log_epoch_metrics: boolean flag to log epoch metrics.

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.loggers.console.ConsoleLogger`
        - :py:mod:`catalyst.loggers.mlflow.MLflowLogger`
        - :py:mod:`catalyst.loggers.neptune.NeptuneLogger`
        - :py:mod:`catalyst.loggers.tensorboard.TensorboardLogger`
    """

    def __init__(self, log_batch_metrics: bool, log_epoch_metrics: bool) -> None:
        self._log_batch_metrics = log_batch_metrics
        self._log_epoch_metrics = log_epoch_metrics

    @property
    def logger(self) -> Any:
        """Internal logger/experiment/etc. from the monitoring system. # noqa: DAR401

        Returns: # noqa: DAR201, DAR202
            Any: internal logger/experiment/etc. from the monitoring system.
        """
        raise NotImplementedError()

    @property
    def log_batch_metrics(self) -> bool:
        """Boolean flag to log batch metrics.

        Returns:
            bool: boolean flag to log batch metrics.
        """
        return self._log_batch_metrics

    @property
    def log_epoch_metrics(self) -> bool:
        """Boolean flag to log epoch metrics.

        Returns:
            bool: boolean flag to log epoch metrics.
        """
        return self._log_epoch_metrics

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
    ) -> None:
        """Logs artifact (arbitrary file like audio, video, etc) to the logger."""
        pass

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to the logger."""
        pass

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyperparameters to the logger."""
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs metrics to the logger."""
        pass

    def flush_log(self) -> None:
        """Flushes the logger."""
        pass

    def close_log(self) -> None:
        """Closes the logger."""
        pass


__all__ = ["ILogger"]
