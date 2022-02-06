from typing import Dict, List, Optional, TYPE_CHECKING
import pickle

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.comet_required:
    import comet_ml
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class CometLogger(ILogger):
    """Comet logger for parameters, metrics, images and other artifacts
    (videos, audio, model checkpoints, etc.).

    You will need a Comet API Key to log your Catalyst runs to Comet.
    You can sign up for a free account here: https://www.comet.ml/signup

    Check out our Quickstart Guide to learn more:
    https://www.comet.ml/docs/quick-start/.

    Args:
        workspace: Workspace to log the experiment.
        project_name: Project to log the experiment.
        experiment_id: Experiment ID of a previously logged Experiment.
            Used to continue logging to an existing experiment (resume experiment).
        comet_mode: Specifies whether to run an Online Experiment
            or Offline Experiment
        tags: A list of tags to add to the Experiment.
        experiment_kwargs: Used to pass additional arguments to
            the Experiment object
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...
            loggers={
                "comet": dl.CometLogger(
                    project_name="my-comet-project"
                )
            }
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "comet": dl.CometLogger(
                        project_name="my-comet-project"
                    )
                }
            # ...

        runner = CustomRunner().run()
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        comet_mode: str = "online",
        tags: List[str] = None,
        logging_frequency: int = 1,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
        **experiment_kwargs: Dict,
    ) -> None:
        super().__init__(
            log_batch_metrics=log_batch_metrics, log_epoch_metrics=log_epoch_metrics
        )
        self.comet_mode = comet_mode
        self.workspace = workspace
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.experiment_kwargs = experiment_kwargs
        self.comet_mode = comet_mode
        self.logging_frequency = logging_frequency

        self.experiment = self._get_experiment(self.comet_mode, self.experiment_id)
        self.experiment.log_other("Created from", "Catalyst")
        if tags is not None:
            self.experiment.add_tags(tags)

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.experiment

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self.experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                workspace=self.workspace,
                project_name=self.project_name,
                **self.experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self.experiment_kwargs,
                )

            return comet_ml.Experiment(
                workspace=self.workspace,
                project_name=self.project_name,
                **self.experiment_kwargs,
            )

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
    ) -> None:
        """Logs artifact (any arbitrary file or object) to the logger."""
        metadata_parameters = {"loader_key": runner.loader_key, "scope": scope}
        passed_metadata_parameters = {
            k: v for k, v in metadata_parameters.items() if v is not None
        }
        if path_to_artifact:
            self.experiment.log_asset(
                path_to_artifact,
                file_name=tag,
                step=runner.batch_step,
                metadata=passed_metadata_parameters,
            )
        else:
            self.experiment.log_asset_data(
                pickle.dumps(artifact),
                file_name=tag,
                step=runner.batch_step,
                epoch=runner.epoch_step,
                metadata=passed_metadata_parameters,
            )

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to the logger."""
        image_name = f"{scope}_{tag}" if scope is not None else tag
        self.experiment.log_image(image, name=image_name, step=runner.batch_step)

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyperparameters to the logger."""
        self.experiment.log_parameters(hparams)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs the metrics to the logger."""
        if (scope == "batch" and not self.log_batch_metrics) or (
            scope in ["loader", "epoch"] and not self.log_epoch_metrics
        ):
            return
        if runner.batch_step % self.logging_frequency == 0:
            self.experiment.log_metrics(
                metrics,
                step=runner.batch_step,
                epoch=runner.epoch_step,
                prefix=f"{runner.loader_key}_{scope}",
            )

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self) -> None:
        """Closes the logger."""
        self.experiment.end()


__all__ = ["CometLogger"]
