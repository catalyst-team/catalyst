from typing import Any, Dict, List, Optional, TYPE_CHECKING
import re

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.mlflow_required:
    import mlflow
    from mlflow.tracking.fluent import ActiveRun
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


def _get_or_start_run(run_name: Optional[str]) -> "ActiveRun":
    """The function of MLflow. Gets the active run and gives it a name.
    If active run does not exist, starts a new one.

    Args:
        run_name: Name of the run

    Returns:
        ActiveRun
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
            raise ValueError(
                f"Unknown type of logging value: type({value})={type(value)}"
            )


class MLflowLogger(ILogger):
    """Mlflow logger for parameters, metrics, images and other artifacts.

    Mlflow documentation: https://mlflow.org/docs/latest/index.html.

    Args:
        experiment: Name of the experiment in MLflow to log to.
        run: Name of the run in Mlflow to log to.
        tracking_uri: URI of tracking server against which
            to log run information related.
        registry_uri: Address of local or remote model registry server.
        exclude: Name of  to exclude from logging.
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).

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
    """

    def __init__(
        self,
        experiment: str,
        run: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
    ) -> None:
        super().__init__(
            log_batch_metrics=log_batch_metrics, log_epoch_metrics=log_epoch_metrics
        )
        self.experiment = experiment
        self.run = run
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.exclude = exclude

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_experiment(self.experiment)
        _get_or_start_run(run_name=self.run)

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return mlflow

    @staticmethod
    def _log_metrics(metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            mlflow.log_metric(f"{key}/{loader_key}{suffix}", value, step=step)

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
    ) -> None:
        """Logs a local file or directory as an artifact to the logger."""
        mlflow.log_artifact(path_to_artifact)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to MLflow for current scope on current step."""
        if scope == "batch" or scope == "loader":
            log_path = "_".join(
                [tag, f"epoch-{runner.epoch_step:04d}", f"loader-{runner.loader_key}"]
            )
        elif scope == "epoch":
            log_path = "_".join([tag, f"epoch-{runner.epoch_step:04d}"])
        elif scope == "experiment" or scope is None:
            log_path = tag
        mlflow.log_image(image, f"{log_path}.png")

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs parameters for current scope.

        Args:
            hparams: Parameters to log.
            runner: experiment runner
        """
        _mlflow_log_params_dict(hparams, log_type="param", exclude=self.exclude)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs batch and epoch metrics to MLflow."""
        if scope == "batch" and self.log_batch_metrics:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics,
                step=runner.batch_step,
                loader_key=runner.loader_key,
                suffix="/batch",
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            for loader_key, per_loader_metrics in metrics.items():
                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=runner.epoch_step,
                    loader_key=loader_key,
                    suffix="/epoch",
                )

    def close_log(self) -> None:
        """End an active MLflow run."""
        mlflow.end_run()


__all__ = ["MLflowLogger"]
