from typing import Any, Dict, Optional

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
from catalyst.typing import Directory, File, Number

if SETTINGS.wandb_required:
    import wandb


class WandbLogger(ILogger):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self.entity = entity
        self.run = wandb.init(
            project=self.project, name=self.name, config=self.config, entity=self.entity
        )

    def _log_metrics(self, metrics, step: int):
        self.run.log(metrics, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
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
        """Logs batch and epoch metrics to wandb."""
        if scope == "batch":
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(metrics=metrics, step=global_epoch_step)
        elif scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=global_epoch_step,
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
        """Logs image to the logger."""
        self.run.log(
            {f"{tag}_scope_{scope}_epoch_{global_epoch_step}.png": wandb.Image(image)},
            step=global_epoch_step,
        )

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        self.run.config.update(hparams)

    def flush_log(self) -> None:
        """Flushes the logger."""
        pass

    def close_log(self) -> None:
        """Closes the logger."""
        self.run.finish()


__all__ = ["WandbLogger"]
