from typing import Dict

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.neptune_required:
    import neptune.new as neptune


class NeptuneLogger(ILogger):
    def __init__(self, base_namespace=None, api_token=None, project=None, run=None, **neptune_run_kwargs):
        self.base_namespace = base_namespace
        self._api_token = api_token
        self._project = project
        self._neptune_run_kwargs = neptune_run_kwargs
        if run is None:
            self.run = neptune.init(project=self._project,
                                    api_token=self._api_token,
                                    **self._neptune_run_kwargs)
        else:
            self.run = run

    @staticmethod
    def _prepare_metrics(metrics):
        conflict_keys = []
        _metrics = {k: v for k, v in metrics.items()}
        for k, v in _metrics.items():
            if k.endswith("/std"):
                k_stripped = k[:-4]
                k_val = k_stripped + "/val"
                if k_val not in _metrics.keys():
                    _metrics[k_val] = _metrics.pop(k_stripped)
        for k in _metrics.keys():
            for j in _metrics.keys():
                if j.startswith(k) and j != k and k not in conflict_keys:
                    conflict_keys.append(k)
        for i in conflict_keys:
            _metrics[i + "_val"] = _metrics.pop(i)
        return _metrics

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            if self.base_namespace is not None:
                self.run[f"{self.base_namespace}/{loader_key}/{suffix}/{key}"].log(value=float(value),
                                                                                   step=step)
            else:
                self.run[f"{loader_key}/{suffix}/{key}"].log(value=float(value),
                                                             step=step)

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
        """Logs batch, epoch and loader metrics to Neptune."""
        if scope == "batch":
            self._log_metrics(
                metrics=metrics, step=global_sample_step, loader_key=loader_key, suffix="batch"
            )
        elif scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                _metrics = self._prepare_metrics(per_loader_metrics)
                self._log_metrics(
                    metrics=_metrics, step=global_epoch_step, loader_key=loader_key, suffix="epoch",
                )
        elif scope == "loader":
            _metrics = self._prepare_metrics(metrics)
            self._log_metrics(
                metrics=_metrics, step=global_epoch_step, loader_key=loader_key, suffix="loader"
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
        if self.base_namespace is not None:
            self.run[f"{self.base_namespace}/{loader_key}/{scope}/epoch_{global_epoch_step}/{tag}"].log(
                neptune.types.File.as_image(image)
            )
        else:
            self.run[f"{loader_key}/{scope}/epoch_{global_epoch_step}/{tag}"].log(
                neptune.types.File.as_image(image)
            )

    def log_hparams(
            self,
            hparams: Dict,
            scope: str = None,
            # experiment info
            run_key: str = None,
            stage_key: str = None,
    ) -> None:
        """Logs hyper-parameters to Neptune."""
        if self.base_namespace is not None:
            self.run[f"{self.base_namespace}/hparams"] = hparams
        else:
            self.run["hparams"] = hparams

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self) -> None:
        """Closes the loggers."""
        self.run.wait()


__all__ = ["NeptuneLogger"]
