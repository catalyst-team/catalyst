from typing import Any, Optional, Dict
import mlflow
import numpy as np
from catalyst.core.logger import ILogger
from catalyst.loggers.functional import EXPERIMENT_PARAMS, STAGE_PARAMS, EXCLUDE_PARAMS, mlflow_log_dict


class MlflowLogger(ILogger):
    """Logger callback, translates ``runner.*_metrics`` to MLflow.
    """
    def __init__(
            self,
            experiment: str,
            run: str,
            tracking_uri: Optional[str] = None,
            registry_uri: Optional[str] = None
    ) -> None:
        """
        Args:
            experiment: Name of the experiment in MLflow to log to.
            run_name: Name of the run in MLflow to log to.
        """
        self.experiment = experiment
        self.run = run
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

        self._multistage = False

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=self.run)

    def _log_metrics(self, metrics: Dict[str, float], step: int, loader_key: str, suffix=""):
        for key, value in metrics.items():
            mlflow.log_metric(f'{key}/{loader_key}{suffix}', value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        scope: str = None,
        # experiment info
        experiment_key: str = None,
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
        """@TODO: docs."""
        if scope == "batch":
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_batch_step, loader_key=loader_key,
                suffix="/batch"
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
        experiment_key: str = None,
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
        """@TODO: docs."""
        mlflow.log_image(image, f'{tag}_{scope}.png')

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        experiment_key: str = None
    ) -> None:
        """@TODO: docs."""
        stages = set(hparams['stages']) - STAGE_PARAMS - EXCLUDE_PARAMS
        self._multistage = len(stages) > 1

        if scope in stages:
            if self._multistage:
                mlflow.start_run(run_name=scope, nested=True)

            scope_params = hparams['stages'].get(scope, {})
            mlflow_log_dict(scope_params, log_type='param')

            for key in STAGE_PARAMS:
                stage_params = hparams['stages'].get(key, {})
                mlflow_log_dict(stage_params, log_type='param')

            for key in EXPERIMENT_PARAMS:
                exp_params = hparams.get(key, {})
                mlflow_log_dict(exp_params, log_type='param')

    def close_log(self) -> None:
        """Finds all running runs and ends them."""
        all_runs = mlflow.search_runs()
        for _ in all_runs[all_runs.status == 'RUNNING']:
            mlflow.end_run()


__all__ = ["MlflowLogger"]
