from typing import Dict, Optional, TYPE_CHECKING
import os
import pickle
import warnings

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.wandb_required:
    import wandb
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class WandbLogger(ILogger):
    """Wandb logger for parameters, metrics, images and other artifacts.

    W&B documentation: https://docs.wandb.com

    Args:
        Project: Name of the project in W&B to log to.
        name: Name of the run in W&B to log to.
        config: Configuration Dictionary for the experiment.
        entity: Name of W&B entity(team) to log to.
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).
        kwargs: Optional,
            additional keyword arguments to be passed directly to the wandb.init

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
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
        **kwargs,
    ) -> None:
        super().__init__(
            log_batch_metrics=log_batch_metrics, log_epoch_metrics=log_epoch_metrics
        )
        if self.log_batch_metrics:
            warnings.warn(
                "Wandb does NOT support several x-axes for logging."
                "For this reason, everything has to be logged in the batch-based regime."
            )

        self.project = project
        self.name = name
        self.entity = entity
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            allow_val_change=True,
            **kwargs,
        )

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.run

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, loader_key: str, prefix=""
    ):
        for key, value in metrics.items():
            self.run.log({f"{key}_{prefix}/{loader_key}": value}, step=step)

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
    ) -> None:
        """Logs artifact (arbitrary file like audio, video, weights) to the logger."""
        if artifact is None and path_to_artifact is None:
            ValueError("Both artifact and path_to_artifact cannot be None")

        artifact = wandb.Artifact(
            name=self.run.id + "_aritfacts",
            type="artifact",
            metadata={"loader_key": runner.loader_key, "scope": scope},
        )

        if artifact:
            art_file_dir = os.path.join("wandb", self.run.id, "artifact_dumps")
            os.makedirs(art_file_dir, exist_ok=True)

            art_file = open(os.path.join(art_file_dir, tag), "wb")
            pickle.dump(artifact, art_file)
            art_file.close()

            artifact.add_file(str(os.path.join(art_file_dir, tag)))
        else:
            artifact.add_file(path_to_artifact)
        self.run.log_artifact(artifact)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to the logger."""
        if scope == "batch" or scope == "loader":
            log_path = "_".join(
                [tag, f"epoch-{runner.epoch_step:04d}", f"loader-{runner.loader}"]
            )
        elif scope == "epoch":
            log_path = "_".join([tag, f"epoch-{runner.epoch_step:04d}"])
        elif scope == "experiment" or scope is None:
            log_path = tag

        step = runner.sample_step if self.log_batch_metrics else runner.epoch_step
        self.run.log({f"{log_path}.png": wandb.Image(image)}, step=step)

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyperparameters to the logger."""
        self.run.config.update(hparams)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs batch and epoch metrics to wandb."""
        step = runner.sample_step if self.log_batch_metrics else runner.epoch_step
        if scope == "batch" and self.log_batch_metrics:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics,
                step=step,
                loader_key=runner.loader_key,
                prefix="batch",
            )
        elif scope == "loader" and self.log_epoch_metrics:
            self._log_metrics(
                metrics=metrics,
                step=step,
                loader_key=runner.loader_key,
                prefix="epoch",
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
            self._log_metrics(
                metrics=per_loader_metrics,
                step=step,
                loader_key=loader_key,
                prefix="epoch",
            )

    def flush_log(self) -> None:
        """Flushes the logger."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the logger."""
        self.run.finish()


__all__ = ["WandbLogger"]
