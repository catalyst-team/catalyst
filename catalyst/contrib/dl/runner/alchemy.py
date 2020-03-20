from typing import Dict  # isort:skip

import warnings

from alchemy import Logger
from deprecation import DeprecatedWarning

from catalyst.dl import utils
from catalyst.dl.core import Experiment, Runner
from catalyst.dl.runner import SupervisedRunner

warnings.simplefilter("always")


class AlchemyRunner(Runner):
    """
    Runner wrapper with Alchemy integration hooks.
    Read about Alchemy here https://alchemy.host
    Powered by Catalyst.Ecosystem

    Example:

        .. code-block:: python

            from catalyst.dl import SupervisedAlchemyRunner

            runner = SupervisedAlchemyRunner()

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                logdir=logdir,
                num_epochs=num_epochs,
                verbose=True,
                monitoring_params={
                    "token": "...", # your Alchemy token
                    "project": "your_project_name",
                    "experiment": "your_experiment_name",
                    "group": "your_experiment_group_name"
                }
            )
    """
    def _init(
        self,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
    ):
        super()._init()
        the_warning = DeprecatedWarning(
            self.__class__.__name__,
            deprecated_in="20.03",
            removed_in="20.04",
            details="Use AlchemyLogger instead."
        )
        warnings.warn(the_warning, category=DeprecationWarning, stacklevel=2)
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if (self.log_on_batch_end and not self.log_on_epoch_end) \
                or (not self.log_on_batch_end and self.log_on_epoch_end):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"

    def _log_metrics(self, metrics: Dict, mode: str, suffix: str = ""):
        for key, value in metrics.items():
            metric_name = f"{key}/{mode}{suffix}"
            self.logger.log_scalar(metric_name, value)

    def _pre_experiment_hook(self, experiment: Experiment):
        monitoring_params = experiment.monitoring_params

        log_on_batch_end: bool = \
            monitoring_params.pop("log_on_batch_end", False)
        log_on_epoch_end: bool = \
            monitoring_params.pop("log_on_epoch_end", True)

        self._init(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
        )
        self.logger = Logger(**monitoring_params)

    def _post_experiment_hook(self, experiment: Experiment):
        self.logger.close()

    def _run_batch(self, batch):
        super()._run_batch(batch=batch)
        if self.log_on_batch_end and not self.state.is_distributed_worker:
            mode = self.state.loader_name
            metrics = self.state.batch_metrics
            self._log_metrics(
                metrics=metrics, mode=mode, suffix=self.batch_log_suffix
            )

    def _run_epoch(self, stage: str, epoch: int):
        super()._run_epoch(stage=stage, epoch=epoch)
        if self.log_on_epoch_end and not self.state.is_distributed_worker:
            mode_metrics = utils.split_dict_to_subdicts(
                dct=self.state.epoch_metrics,
                prefixes=list(self.state.loaders.keys()),
                extra_key="_base",
            )
            for mode, metrics in mode_metrics.items():
                self._log_metrics(
                    metrics=metrics, mode=mode, suffix=self.epoch_log_suffix
                )

    def run_experiment(self, experiment: Experiment):
        """Starts experiment

        Args:
            experiment (Experiment): experiment class
        """
        self._pre_experiment_hook(experiment=experiment)
        super().run_experiment(experiment=experiment)
        self._post_experiment_hook(experiment=experiment)


class SupervisedAlchemyRunner(AlchemyRunner, SupervisedRunner):
    """SupervisedRunner with Alchemy"""
    pass


__all__ = ["AlchemyRunner", "SupervisedAlchemyRunner"]
