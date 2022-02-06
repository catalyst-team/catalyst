from typing import Dict, TYPE_CHECKING
import os

import numpy as np

from tensorboardX import SummaryWriter
import torch

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Creates tensor from RGB image.

    Args:
        image: RGB image stored as np.ndarray

    Returns:
        tensor
    """
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


class TensorboardLogger(ILogger):
    """Tensorboard logger for parameters, metrics, images and other artifacts.

    Args:
        logdir: path to logdir for tensorboard.
        use_logdir_postfix: boolean flag
            to use extra ``tensorboard`` prefix in the logdir.
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).

    .. note::
        This logger is used by default by ``dl.Runner`` and ``dl.SupervisedRunner``
        in case of specified logdir during ``runner.train(..., logdir=/path/to/logdir)``.

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"tensorboard": dl.TensorboardLogger(logdir="./logdir/tensorboard"}
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "tensorboard": dl.TensorboardLogger(logdir="./logdir/tensorboard")
                }

            # ...

        runner = CustomRunner().run()
    """

    def __init__(
        self,
        logdir: str,
        use_logdir_postfix: bool = False,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
    ):
        """Init."""
        super().__init__(
            log_batch_metrics=log_batch_metrics, log_epoch_metrics=log_epoch_metrics
        )
        if use_logdir_postfix:
            logdir = os.path.join(logdir, "tensorboard")
        self.logdir = logdir
        self.loggers = {}
        os.makedirs(self.logdir, exist_ok=True)

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.loggers

    def _check_loader_key(self, loader_key: str):
        if loader_key not in self.loggers.keys():
            logdir = os.path.join(self.logdir, f"{loader_key}")
            self.loggers[loader_key] = SummaryWriter(logdir)

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""
    ):
        for key, value in metrics.items():
            self.loggers[loader_key].add_scalar(f"{key}{suffix}", float(value), step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to Tensorboard for current scope on current step."""
        assert runner.loader_key is not None
        self._check_loader_key(loader_key=runner.loader_key)
        tensor = _image_to_tensor(image)
        self.loggers[runner.loader_key].add_image(
            f"{tag}", tensor, global_step=runner.epoch_step
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs batch and epoch metrics to Tensorboard."""
        if scope == "batch" and self.log_batch_metrics:
            self._check_loader_key(loader_key=runner.loader_key)
            # metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics,
                step=runner.sample_step,
                loader_key=runner.loader_key,
                suffix="/batch",
            )
        elif scope == "loader" and self.log_epoch_metrics:
            self._check_loader_key(loader_key=runner.loader_key)
            self._log_metrics(
                metrics=metrics,
                step=runner.epoch_step,
                loader_key=runner.loader_key,
                suffix="/epoch",
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            # @TODO: remove naming magic
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
            self._check_loader_key(loader_key=loader_key)
            self._log_metrics(
                metrics=per_loader_metrics,
                step=runner.epoch_step,
                loader_key=loader_key,
                suffix="/epoch",
            )

    def flush_log(self) -> None:
        """Flushes the loggers."""
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        """Closes the loggers."""
        for logger in self.loggers.values():
            logger.close()


__all__ = ["TensorboardLogger"]
