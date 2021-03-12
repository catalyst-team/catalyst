from typing import Dict
import os

import numpy as np
from tensorboardX import SummaryWriter

from catalyst.contrib.utils.visualization import image_to_tensor
from catalyst.core.logger import ILogger


class TensorboardLogger(ILogger):
    """Logger callback, translates ``runner.metric_manager`` to tensorboard."""

    def __init__(self, logdir: str, use_logdir_postfix: bool = False):
        """@TODO: docs."""
        if use_logdir_postfix:
            logdir = os.path.join(logdir, "tensorboard")
        self.logdir = logdir
        self.loggers = {}
        os.makedirs(self.logdir, exist_ok=True)

    def _check_loader_key(self, loader_key: str):
        if loader_key not in self.loggers.keys():
            logdir = os.path.join(self.logdir, f"{loader_key}")
            self.loggers[loader_key] = SummaryWriter(logdir)

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            self.loggers[loader_key].add_scalar(f"{key}{suffix}", value, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        experiment_key: str = None,
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
        """@TODO: docs."""
        if scope == "batch":
            self._check_loader_key(loader_key=loader_key)
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_batch_step, loader_key=loader_key, suffix="/batch"
            )
        elif scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                self._check_loader_key(loader_key=loader_key)
                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=global_epoch_step,
                    loader_key=loader_key,
                    suffix="/epoch",
                )

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        scope: str = None,
        # experiment info
        experiment_key: str = None,
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
        """@TODO: docs."""
        if scope == "loader":
            self._check_loader_key(loader_key=loader_key)
            tensor = image_to_tensor(image)
            self.loggers[loader_key].add_image(f"{tag}", tensor, global_step=global_epoch_step)

    def flush_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.close()


__all__ = ["TensorboardLogger"]
