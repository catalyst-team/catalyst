# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, TYPE_CHECKING, Any

import mlflow

from catalyst.core.callback import (
    Callback,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
)
from catalyst.utils.misc import split_dict_to_subdicts

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class MLFlowLogger(Callback):

    def __init__(
            self,
            experiment: str,
            metric_names: List[str] = None,
            log_on_batch_end: bool = False,
            log_on_epoch_end: bool = True,
            tracking_uri: str = None,
            registry_uri: str = None,
            params: Dict[str, Any] = None
    ):
        """
        Args:
            experiment: name of experiment
            metric_names: list of metric names to log,
                if None - logs everything
            log_on_batch_end: logs per-batch metrics if set True
            log_on_epoch_end: logs per-epoch metrics if set True
            tracking_uri: Tracking URI address
            registry_uri: Registry URI address
            params: parameters to be logged in MLFlow
        """
        super().__init__(
            order=CallbackOrder.logging,
            node=CallbackNode.master,
            scope=CallbackScope.experiment,
        )
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        if (self.log_on_batch_end and not self.log_on_epoch_end) or (
                not self.log_on_batch_end and self.log_on_epoch_end
        ):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"

        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment = experiment

        self.params = dict() if params is None else params

    def _log_metrics(
            self,
            metrics: Dict[str, float],
            step: int,
            mode: str,
            suffix="",
    ):
        if self.metrics_to_log is None:
            metrics_to_log = set(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        metrics = {
            f"{key}/{mode}{suffix}": value
            for key, value in metrics.items()
            if key in metrics_to_log
        }
        mlflow.log_metrics(metrics, step=step)

    def on_stage_start(self, runner: "IRunner"):
        """Initialize MLFlow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run()
        mlflow.log_params(self.params)

    def on_batch_end(self, runner: "IRunner"):
        """Translate batch metrics to MLFlow."""
        if self.log_on_batch_end:
            mode = runner.loader_key
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix=self.batch_log_suffix,
            )

    def on_loader_end(self, runner: "IRunner"):
        """Translate loader metrics to MLFlow."""
        if self.log_on_epoch_end:
            mode = runner.loader_key
            metrics = runner.loader_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_epoch,
                mode=mode,
                suffix=self.epoch_log_suffix,
            )

    def on_epoch_end(self, runner: "IRunner"):
        """Translate epoch metrics to Weights & Biases."""
        extra_mode = "_base"
        splitted_epoch_metrics = split_dict_to_subdicts(
            dct=runner.epoch_metrics,
            prefixes=list(runner.loaders.keys()),
            extra_key=extra_mode,
        )

        if self.log_on_epoch_end:
            if extra_mode in splitted_epoch_metrics.keys():
                # if we are using OptimizerCallback
                self._log_metrics(
                    metrics=splitted_epoch_metrics[extra_mode],
                    step=runner.global_epoch,
                    mode=extra_mode,
                    suffix=self.epoch_log_suffix,
                )

    def on_stage_end(self, runner: "IRunner"):
        """Finish logging to MLFlow."""
        mlflow.end_run()


__all__ = ["MLFlowLogger"]
