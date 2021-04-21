from typing import Dict

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.neptune_required:
    import neptune.new as neptune


class NeptuneLogger(ILogger):
    def __init__(self, base_namespace=None, api_token=None, project=None, run=None, **neptune_run_kwargs):
        if base_namespace is None:
            self.base_namespace = 'catalyst_training'
        else:
            self.base_namespace = base_namespace
        self._api_token = api_token
        self._project = project
        self._neptune_run_kwargs = neptune_run_kwargs
        if run is None:
            self.run = neptune.init(project=self._project, api_token=self._api_token, run=run,
                                    **self._neptune_run_kwargs)

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            self.run[f"{loader_key}/{key}/{suffix}"].log(value=float(value), step=step)

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
        """Logs batch and epoch metrics to Neptune."""
        if scope == "batch":
            # metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_sample_step, loader_key=loader_key, suffix="batch"
            )
        elif scope == "epoch":
            self._log_metrics(
                metrics=metrics, step=global_epoch_step, loader_key=loader_key, suffix="epoch",
            )
        # ToDo check below logic
        elif scope == "loader":
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
            self._log_metrics(
                metrics=per_loader_metrics,
                step=global_epoch_step,
                loader_key=loader_key,
                suffix="loader",
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
        """Logs image to Neptune for current scope on current step."""
        assert loader_key is not None
        self.run[f"{loader_key}/{tag}/{scope}/epoch_{global_epoch_step}"].log(neptune.types.File.as_image(image))

    def log_hparams(
            self,
            hparams: Dict,
            scope: str = None,
            # experiment info
            run_key: str = None,
            stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to Neptune."""
        self.run.config.update(hparams)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        self.run.wait()

    def close_log(self) -> None:
        """Closes the loggers."""
        self.run.wait()


__all__ = ["NeptuneLogger"]
