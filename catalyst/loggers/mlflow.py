from typing import Any, Dict, List, Optional
import re

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
from catalyst.typing import Directory, File, Union

if SETTINGS.mlflow_required:
    import mlflow


def _get_or_start_run(run_name):
    """The function of MLflow. Gets the active run and gives it a name.
    If active run does not exist, starts a new one.
    """
    active_run = mlflow.active_run()
    if active_run:
        mlflow.set_tag("mlflow.runName", run_name)
        return active_run
    return mlflow.start_run(run_name=run_name)


def _mlflow_log_params_dict(
    dictionary: Dict[str, Any],
    prefix: Optional[str] = None,
    log_type: Optional[str] = None,
    exclude: Optional[List[str]] = None,
):
    """The function of MLflow. Logs any value by its type from dictionary recursively.

    Args:
        dictionary: Values to log as dictionary.
        prefix: Prefix for parameter name (if the parameter is composite).
        log_type: The entity of logging (param, metric, artifact, image, etc.).
        exclude: Keys in the dictionary to exclude from logging.

    Raises:
        ValueError: If meets unknown type or log_type for logging in MLflow
            (add new case if needed).
    """
    for name, value in dictionary.items():
        if exclude is not None and name in exclude:
            continue

        name = re.sub(r"\W", "", name)
        name = f"{prefix}/{name}" if prefix else name

        if log_type == "dict":
            mlflow.log_dict(dictionary, name)
        elif isinstance(value, dict):
            _mlflow_log_params_dict(value, name, log_type, exclude)
        elif log_type == "param":
            try:
                mlflow.log_param(name, value)
            except mlflow.exceptions.MlflowException:
                continue
        else:
            raise ValueError(f"Unknown type of logging value: type({value})={type(value)}")


class MLflowLogger(ILogger):
    """Mlflow logger for parameters, metrics, images and other artifacts.

    Mlflow documentation: https://mlflow.org/docs/latest/index.html.

    Args:
        experiment: Name of the experiment in MLflow to log to.
        run: Name of the run in Mlflow to log to.
        tracking_uri: URI of tracking server against which to log run information related.
        registry_uri: Address of local or remote model registry server.
        exclude: Name of  to exclude from logging.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"mlflow": dl.MLflowLogger(experiment="test_exp", run="test_run")}
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "mlflow": dl.MLflowLogger(experiment="test_exp", run="test_run")
                }

            # ...

        runner = CustomRunner().run()

    Config API example:

    .. code-block:: yaml

        loggers:
            mlflow:
                _target_: MLflowLogger
                experiment: test_exp
                run: test_run
        ...

    Hydra API example:

    .. code-block:: yaml

        loggers:
            mlflow:
                _target_: catalyst.dl.MLflowLogger
                experiment: test_exp
                run: test_run
        ...
    """

    def __init__(
        self,
        experiment: str,
        run: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        exclude: Optional[List[str]] = None,
    ) -> None:
        self.experiment = experiment
        self.run = run
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.exclude = exclude

        self._multistage = False

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_experiment(self.experiment)
        _get_or_start_run(run_name=self.run)

    @staticmethod
    def _log_metrics(metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            mlflow.log_metric(f"{key}/{loader_key}{suffix}", value, step=step)

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
        """Logs batch and epoch metrics to MLflow."""
        if scope == "batch":
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_batch_step, loader_key=loader_key, suffix="/batch"
            )
        elif scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
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
        """Logs image to MLflow for current scope on current step."""
        mlflow.log_image(image, f"{tag}_scope_{scope}_epoch_{global_epoch_step}.png")

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs parameters for current scope.

        If there in experiment more than one stage, creates nested runs.

        Note:
            If the scope is "experiment", it does nothing, since overwriting parameters
            in MLflow is prohibited. Thus, first, the parameters of the stage
            are recorded, and only then the experiment.

        Args:
            hparams: Parameters to log.
            scope: On which scope log parameters.
            run_key: Experiment info.
            stage_key: Stage info.
        """
        stages = hparams.get("stages", {})
        self._multistage = len(stages) > 1

        if scope == "experiment":
            if not self.run:
                mlflow.set_tag("mlflow.runName", run_key)

        if scope == "stage":
            if self._multistage:
                mlflow.start_run(run_name=stage_key, nested=True)

            stage_params = hparams.get("stages", {}).get(stage_key, {})
            _mlflow_log_params_dict(stage_params, log_type="param", exclude=self.exclude)

            experiment_params = {key: value for key, value in hparams.items() if key != "stages"}
            _mlflow_log_params_dict(experiment_params, log_type="param", exclude=self.exclude)

    def log_artifact(
        self,
        tag: str,
        artifact: Union[Directory, File] = None,
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
        """Logs a local file or directory as an artifact to the logger."""
        mlflow.log_artifact(path_to_artifact)

    def close_log(self, scope: str = None) -> None:
        """End an active MLflow run."""
        if scope == "stage" and self._multistage:
            mlflow.end_run()
        if scope == "experiment":
            mlflow.end_run()


__all__ = ["MLflowLogger"]
