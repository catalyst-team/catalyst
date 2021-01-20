# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, TYPE_CHECKING

from alchemy import Logger

from catalyst import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder, CallbackScope

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class AlchemyLogger(Callback):
    """Logger callback, translates ``runner.*_metrics`` to Alchemy.
    Read about Alchemy here https://alchemy.host

    Example:
        .. code-block:: python

            from catalyst.dl import SupervisedRunner, AlchemyLogger

            runner = SupervisedRunner()

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                logdir=logdir,
                num_epochs=num_epochs,
                verbose=True,
                callbacks={
                    "logger": AlchemyLogger(
                        token="...", # your Alchemy token
                        project="your_project_name",
                        experiment="your_experiment_name",
                        group="your_experiment_group_name",
                    )
                }
            )

    Powered by Catalyst.Ecosystem.
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
        **logging_params,
    ):
        """
        Args:
            metric_names: list of metric names to log,
                if none - logs everything
            log_on_batch_end: logs per-batch metrics if set True
            log_on_epoch_end: logs per-epoch metrics if set True
        """
        super().__init__(
            order=CallbackOrder.logging, node=CallbackNode.master, scope=CallbackScope.experiment,
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

        self.logger = Logger(**logging_params)

    def __del__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.logger.close()

    def _log_metrics(self, metrics: Dict[str, float], step: int, mode: str, suffix=""):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                metric_name = f"{name}/{mode}{suffix}"
                metric_value = metrics[name]
                self.logger.log_scalar(
                    name=metric_name, value=metric_value, step=step,
                )

    def on_batch_end(self, runner: "IRunner"):
        """Translate batch metrics to Alchemy."""
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
        """Translate loader metrics to Alchemy."""
        if self.log_on_epoch_end:
            mode = runner.loader_key
            metrics = runner.loader_metrics
            self._log_metrics(
                metrics=metrics, step=runner.global_epoch, mode=mode, suffix=self.epoch_log_suffix,
            )

    def on_epoch_end(self, runner: "IRunner"):
        """Translate epoch metrics to Alchemy."""
        extra_mode = "_base"
        splitted_epoch_metrics = utils.split_dict_to_subdicts(
            dct=runner.epoch_metrics, prefixes=list(runner.loaders.keys()), extra_key=extra_mode,
        )

        if self.log_on_epoch_end:
            self._log_metrics(
                metrics=splitted_epoch_metrics[extra_mode],
                step=runner.global_epoch,
                mode=extra_mode,
                suffix=self.epoch_log_suffix,
            )


__all__ = ["AlchemyLogger"]
