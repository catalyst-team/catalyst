from typing import Dict, List

from alchemy import Logger

from catalyst import utils
from catalyst.core import (
    Callback,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
    State,
)


class AlchemyLogger(Callback):
    """Logger callback, translates ``state.*_metrics`` to Alchemy.
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
                        "token": "...", # your Alchemy token
                        "project": "your_project_name",
                        "experiment": "your_experiment_name",
                        "group": "your_experiment_group_name",
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
            metric_names (List[str]): list of metric names to log,
                if none - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
        """
        super().__init__(
            order=CallbackOrder.Logging,
            node=CallbackNode.Master,
            scope=CallbackScope.Experiment,
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
                self.logger.log_scalar(
                    name=metric_name, value=metric_value, step=step,
                )

    def on_batch_end(self, state: State):
        """Translate batch metrics to Alchemy."""
        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.batch_metrics
            self._log_metrics(
                metrics=metrics_,
                step=state.global_step,
                mode=mode,
                suffix=self.batch_log_suffix,
            )

    def on_loader_end(self, state: State):
        """Translate loader metrics to Alchemy."""
        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.loader_metrics
            self._log_metrics(
                metrics=metrics_,
                step=state.global_epoch,
                mode=mode,
                suffix=self.epoch_log_suffix,
            )

    def on_epoch_end(self, state: State):
        """Translate epoch metrics to Alchemy."""
        extra_mode = "_base"
        splitted_epoch_metrics = utils.split_dict_to_subdicts(
            dct=state.epoch_metrics,
            prefixes=list(state.loaders.keys()),
            extra_key=extra_mode,
        )

        if self.log_on_epoch_end:
            self._log_metrics(
                metrics=splitted_epoch_metrics[extra_mode],
                step=state.global_epoch,
                mode=extra_mode,
                suffix=self.epoch_log_suffix,
            )
