from typing import Any, Dict

import numpy as np


class ILogger:
    """An abstraction that syncs experiment run with monitoring tools.

    Args:
        log_batch_metrics: boolean flag to log batch metrics.
        log_epoch_metrics: boolean flag to log epoch metrics.

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.loggers.console.ConsoleLogger`
        - :py:mod:`catalyst.loggers.tensorboard.TensorboardLogger`
        - :py:mod:`catalyst.loggers.mlflow.MLflowLogger`
        - :py:mod:`catalyst.loggers.neptune.NeptuneLogger`
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
        raise NotImplementedError

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

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs metrics to the logger."""
        pass

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs image to the logger."""
        pass

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        pass

    def log_artifact(
        self,
        tag: str,
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs artifact (arbitrary file like audio, video, model weights) to the logger."""
        pass

    def flush_log(self) -> None:
        """Flushes the logger."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the logger."""
        pass


__all__ = ["ILogger"]
