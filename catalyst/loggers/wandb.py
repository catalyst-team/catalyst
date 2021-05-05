from typing import Any, Dict, Optional

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.wandb_required:
    import wandb


class WandbLogger(ILogger):
    """Wandb logger for parameters, metrics, images and other artifacts.

    W&B documentation: https://docs.wandb.com

    Args:
        Project: Name of the project in W&B to log to.
        name: Name of the run in W&B to log to.
        config: Configuration Dictionary for the experiment.
        entity: Name of W&B entity(team) to log to.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"wandb": dl.WandbLogger(project="wandb_test", name="expeirment_1")}
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "wandb": dl.WandbLogger(project="wandb_test", name="experiment_1")
                }

            # ...

        runner = CustomRunner().run()

    Config API example:

    .. code-block:: yaml

        loggers:
            wandb:
                _target_: WandbLogger
                project: test_exp
                name: test_run
        ...

    Hydra API example:

    .. code-block:: yaml

        loggers:
            wandb:
                _target_: catalyst.dl.WandbLogger
                project: test_exp
                name: test_run
        ...
    """

    def __init__(
        self, project: str, name: Optional[str] = None, entity: Optional[str] = None,
    ) -> None:
        self.project = project
        self.name = name
        self.entity = entity
        self.run = wandb.init(
            project=self.project, name=self.name, entity=self.entity, allow_val_change=True
        )

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, prefix=""):
        for key, value in metrics.items():
            self.run.log({f"{key}_{prefix}/{loader_key}": value}, step=step)

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
            self._log_metrics(
                metrics=metrics, step=global_epoch_step, loader_key=loader_key, prefix="batch"
            )
        elif scope == "loader":
            self._log_metrics(
                metrics=metrics, step=global_epoch_step, loader_key=loader_key, prefix="epoch",
            )
        elif scope == "epoch":
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
            self._log_metrics(
                metrics=per_loader_metrics,
                step=global_epoch_step,
                loader_key=loader_key,
                prefix="epoch",
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
