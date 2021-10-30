from typing import Dict, List, Optional
import pickle

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.comet_required:
    import comet_ml


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

    Config API example:

    .. code-block:: yaml

        loggers:
            comet:
                _target_: CometLogger
                project_name: my_comet_project
        ...

    Hydra API example:

    .. code-block:: yaml

        loggers:
            comet:
                _target_: catalyst.dl.CometLogger
                project_name: my_comet_project
        ...

    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        comet_mode: str = "online",
        tags: List[str] = None,
        logging_frequency: int = 1,
        **experiment_kwargs: Dict,
    ) -> None:
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
                workspace=self.workspace, project_name=self.project_name, **self.experiment_kwargs
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
                workspace=self.workspace, project_name=self.project_name, **self.experiment_kwargs
            )

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
        """Logs the metrics to the logger."""
        if global_batch_step % self.logging_frequency == 0:
            self.experiment.log_metrics(
                metrics,
                step=global_batch_step,
                epoch=global_epoch_step,
                prefix=f"{stage_key}/{loader_key}_{scope}",
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
        image_name = f"{scope}_{tag}"

        self.experiment.log_image(image, name=image_name, step=global_batch_step)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        self.experiment.log_parameters(hparams, prefix=scope)

    def log_artifact(
        self,
        tag: str = "artifact",
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
        """Logs artifact (any arbitrary file or object) to the logger."""
        metadata_parameters = {
            "stage_key": stage_key,
            "loader_key": loader_key,
            "scope": scope,
        }
        passed_metadata_parameters = {
            k: v for k, v in metadata_parameters.items() if v is not None
        }
        if path_to_artifact:
            self.experiment.log_asset(
                path_to_artifact,
                file_name=tag,
                step=global_batch_step,
                metadata=passed_metadata_parameters,
            )
        else:
            self.experiment.log_asset_data(
                pickle.dumps(artifact),
                file_name=tag,
                step=global_batch_step,
                epoch=global_epoch_step,
                metadata=passed_metadata_parameters,
            )

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the logger."""
        if scope is None or scope == "experiment":
            self.experiment.end()


__all__ = ["CometLogger"]
