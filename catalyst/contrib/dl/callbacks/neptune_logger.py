# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List

import neptune

from catalyst.core.callback import (
    Callback,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
)
from catalyst.core.runner import IRunner


class NeptuneLogger(Callback):
    """Logger callback, translates ``runner.*_metrics`` to Neptune.
    Read about Neptune here https://neptune.ai

    Example:
        .. code-block:: python

            from catalyst.dl import SupervisedRunner
            from catalyst.contrib.dl.callbacks.neptune import NeptuneLogger

            runner = SupervisedRunner()

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                logdir=logdir,
                num_epochs=num_epochs,
                verbose=True,
                callbacks=[
                    NeptuneLogger(
                        api_token="...", # your Neptune token
                        project_name="your_project_name",
                        offline_mode=False, # turn off neptune for debug
                        name="your_experiment_name",
                        params={...},  # your hyperparameters
                        tags=["resnet", "no-augmentations"], # tags
                        upload_source_files=["*.py"], # files to save
                         )
                    ]
                 )

        You can see an example experiment here:
        https://ui.neptune.ai/o/shared/org/catalyst-integration/e/CAT-13/charts

        You can log your experiments without registering.
        Just use "ANONYMOUS" token::

            runner.train(
                ...
                callbacks=[
                    "NepuneLogger(
                        api_token="ANONYMOUS",
                        project_name="shared/catalyst-integration",
                        ...
                         )
                    ]
                )
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
        offline_mode: bool = False,
        **logging_params,
    ):
        """
        Args:
            metric_names (List[str]): list of metric names to log,
                if none - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
            offline_mode (bool): whether logging to Neptune server should
                 be turned off. It is useful for debugging
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

        if offline_mode:
            neptune.init(
                project_qualified_name="dry-run/project",
                backend=neptune.OfflineBackend(),
            )
        else:
            neptune.init(
                api_token=logging_params["api_token"],
                project_qualified_name=logging_params["project_name"],
            )

        logging_params.pop("api_token")
        logging_params.pop("project_name")

        self.experiment = neptune.create_experiment(**logging_params)

    def __del__(self):
        """@TODO: Docs. Contribution is welcome"""
        if hasattr(self, "experiment"):
            self.experiment.stop()

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                metric_name = f"{name}/{mode}{suffix}"
                metric_value = metrics[name]
                self.experiment.log_metric(metric_name, y=metric_value, x=step)

    def on_batch_end(self, runner: IRunner):
        """Log batch metrics to Neptune."""
        if self.log_on_batch_end:
            mode = runner.loader_name
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix=self.batch_log_suffix,
            )

    def on_loader_end(self, runner: IRunner):
        """Translate epoch metrics to Neptune."""
        if self.log_on_epoch_end:
            mode = runner.loader_name
            metrics = runner.loader_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_epoch,
                mode=mode,
                suffix=self.epoch_log_suffix,
            )


__all__ = ["NeptuneLogger"]
