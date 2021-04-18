from typing import Dict
import os

import numpy as np
from tensorboardX import SummaryWriter

from catalyst.core.logger import ILogger
from catalyst.loggers.functional import image_to_tensor


class TensorboardLogger(ILogger):
    """Tensorboard logger for parameters, metrics, images and other artifacts.

    Args:
        logdir: path to logdir for tensorboard
        use_logdir_postfix: boolean flag to use extra ``tensorboard`` prefix in the logdir

    .. note::
        This logger is used by default by ``dl.Runner`` and ``dl.SupervisedRunner`` in case of
        specified logdir during ``runner.train(..., logdir=/path/to/logdir)``.

    .. note::
        This logger is used by default by ``dl.ConfigRunner`` and ``dl.HydraRunner`` in case of
        specified logdir in config ``args``.

    Notebook API examples:

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

    Config API example:

    .. code-block:: yaml

        loggers:
            tensorboard:
                _target_: TensorboardLogger
                logdir: ./logdir/tensorboard
        ...

    Hydra API example:

    .. code-block:: yaml

        loggers:
            tensorboard:
                _target_: catalyst.dl.TensorboardLogger
                logdir: ./logdir/tensorboard
        ...
    """

    def __init__(self, logdir: str, use_logdir_postfix: bool = False):
        """Init."""
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
            self.loggers[loader_key].add_scalar(f"{key}{suffix}", float(value), step)

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
        """Logs batch and epoch metrics to Tensorboard."""
        if scope == "batch":
            self._check_loader_key(loader_key=loader_key)
            # metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_sample_step, loader_key=loader_key, suffix="/batch"
            )
        elif scope == "loader":
            self._check_loader_key(loader_key=loader_key)
            self._log_metrics(
                metrics=metrics, step=global_epoch_step, loader_key=loader_key, suffix="/epoch",
            )
        elif scope == "epoch":
            # @TODO: remove naming magic
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
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
        """Logs image to Tensorboard for current scope on current step."""
        assert loader_key is not None
        self._check_loader_key(loader_key=loader_key)
        tensor = image_to_tensor(image)
        self.loggers[loader_key].add_image(f"{tag}/{scope}", tensor, global_step=global_epoch_step)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        for logger in self.loggers.values():
            logger.flush()

    def close_log(self) -> None:
        """Closes the loggers."""
        for logger in self.loggers.values():
            logger.close()


__all__ = ["TensorboardLogger"]
