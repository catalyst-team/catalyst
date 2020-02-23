from typing import Dict  # isort:skip

from alchemy import Logger

from catalyst.dl import utils
from catalyst.dl.core import Experiment, Runner
from catalyst.dl.runner import SupervisedRunner


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
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.is_distributed_worker = utils.get_rank() > 0

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
        if self.log_on_batch_end and not self.is_distributed_worker:
            mode = self.state.loader_name
            metrics = self.state.metric_manager.batch_values
            self._log_metrics(
                metrics=metrics, mode=mode, suffix=self.batch_log_suffix
            )

    def _run_epoch(self, stage: str, epoch: int):
        super()._run_epoch(stage=stage, epoch=epoch)
        if self.log_on_epoch_end and not self.is_distributed_worker:
            for mode, metrics in \
                    self.state.metric_manager.epoch_values.items():
                self._log_metrics(
                    metrics=metrics, mode=mode, suffix=self.epoch_log_suffix
                )

    def run_experiment(self, experiment: Experiment, check: bool = False):
        """Starts experiment

        Args:
            experiment (Experiment): experiment class
            check (bool): if ``True`` takes only 3 steps
        """
        self._pre_experiment_hook(experiment=experiment)
        super().run_experiment(experiment=experiment, check=check)
        self._post_experiment_hook(experiment=experiment)


class SupervisedAlchemyRunner(AlchemyRunner, SupervisedRunner):
    """SupervisedRunner with Alchemy"""
    pass


__all__ = ["AlchemyRunner", "SupervisedAlchemyRunner"]
