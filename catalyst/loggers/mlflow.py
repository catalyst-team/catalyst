from typing import Any, Dict, Optional

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
from catalyst.typing import Directory, File, Number

if SETTINGS.mlflow_required:
    import mlflow

EXPERIMENT_PARAMS = (
    "shared",
    "args",
    "runner",
    "engine",
    "model",
    "stages",
)
STAGE_PARAMS = ("data", "criterion", "optimizer", "scheduler", "stage")
EXCLUDE_PARAMS = ("loggers", "transform", "callbacks")


def _get_or_start_run(run_name):
    """The function of MLflow. Gets the active run and gives it a name.
    If active run does not exist, starts a new one.
    """
    active_run = mlflow.active_run()
    if active_run:
        mlflow.set_tag("mlflow.runName", run_name)
        return active_run
    return mlflow.start_run(run_name=run_name)


def _mlflow_log_dict(dictionary: Dict[str, Any], prefix: str = "", log_type: Optional[str] = None):
    """The function of MLflow. Logs any value by its type from dictionary recursively.

    Args:
        dictionary: Values to log as dictionary.
        prefix: Prefix for parameter name (if the parameter is composite).
        log_type: The entity of logging (param, metric, artifact, image, etc.).

    Raises:
        ValueError: If meets unknown type or log_type for logging in MLflow
            (add new case if needed).
    """
    for name, value in dictionary.items():
        if name in EXCLUDE_PARAMS:
            continue

        name = name.replace("*", "")
        if prefix not in STAGE_PARAMS and prefix:
            name = f"{prefix}/{name}"

        if log_type == "dict":
            mlflow.log_dict(dictionary, name)
        elif isinstance(value, dict):
            _mlflow_log_dict(value, name, log_type)
        elif log_type == "param":
            try:
                mlflow.log_param(name, value)
            except mlflow.exceptions.MlflowException:
                continue
        elif isinstance(value, (Directory, File)) or log_type == "artifact":
            mlflow.log_artifact(value)
        elif isinstance(value, Number):
            mlflow.log_metric(name, value)
        else:
            raise ValueError(f"Unknown type of logging value: {type(value)}")


class MLflowLogger(ILogger):
    """Mlflow logger for parameters, metrics, images and other artifacts.

    Mlflow documentation: https://mlflow.org/docs/latest/index.html.

    Args:
        experiment: Name of the experiment in MLflow to log to.
        run: Name of the run in Mlflow to log to.
        tracking_uri: URI of tracking server against which to log run information related.
        registry_uri: Address of local or remote model registry server.

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
    ) -> None:
        self.experiment = experiment
        self.run = run
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

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

        If the scope is "experiment", it does nothing, since overwriting parameters in
        MLflow is prohibited. Thus, first, the parameters of the stage are recorded,
        and only then the experiment.

        Args:
            hparams: Parameters to log.
            scope: On which scope log parameters.
            run_key: Experiment info.
            stage_key: Stage info.
        """
        stages = set(hparams.get("stages", {})) - set(STAGE_PARAMS) - set(EXCLUDE_PARAMS)
        self._multistage = len(stages) > 1

        if scope == "experiment":
            if self._multistage:
                mlflow.set_tag("mlflow.runName", run_key)

        if scope == "stage":
            if self._multistage:
                mlflow.start_run(run_name=stage_key, nested=True)

            scope_params = hparams.get("stages", {}).get(run_key, {})
            _mlflow_log_dict(scope_params, log_type="param")

            for key in STAGE_PARAMS:
                stage_params = hparams.get("stages", {}).get(key, {})
                _mlflow_log_dict(stage_params, log_type="param")

            for key in EXPERIMENT_PARAMS:
                exp_params = hparams.get(key, {})
                _mlflow_log_dict(exp_params, log_type="param")

    def close_log(self) -> None:
        """Finds all **running** runs and ends them."""
        all_runs = mlflow.search_runs()
        for _ in all_runs[all_runs.status == "RUNNING"]:
            mlflow.end_run()


__all__ = ["MLflowLogger"]
