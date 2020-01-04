from typing import Tuple  # isort:skip
import os
from pathlib import Path

from catalyst.dl import utils
from catalyst.utils.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)
from catalyst.core import Runner
from .experiment import Experiment
from .state import DLRunnerState


class DLRunner(Runner):
    """
    Abstract class for all runners inherited from
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.experiment: Experiment = None

    def _get_experiment_components(
        self, stage: str = None
    ) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        utils.set_global_seed(self.experiment.initial_seed)
        model = self.experiment.get_model(stage)
        criterion, optimizer, scheduler = \
            self.experiment.get_experiment_components(model, stage)

        model, criterion, optimizer, scheduler, device = \
            utils.process_components(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                distributed_params=self.experiment.distributed_params,
                device=self.device
            )

        return model, criterion, optimizer, scheduler, device

    def _prepare_for_stage(self, stage: str):
        utils.set_global_seed(self.experiment.initial_seed)
        migrating_params = {}
        if self.state is not None:
            migrating_params.update(
                {
                    "step": self.state.step,
                    "epoch": self.state.epoch
                }
            )

        self.model, criterion, optimizer, scheduler, self.device = \
            self._get_experiment_components(stage)

        self.state = DLRunnerState(
            stage=stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **self.experiment.get_state_params(stage),
            **migrating_params
        )
        utils.set_global_seed(self.experiment.initial_seed)

        # @TODO: remove this trick
        loaders = self.experiment.get_loaders(stage)
        self.loaders = loaders
        callbacks = self.experiment.get_callbacks(stage)
        utils.set_global_seed(self.experiment.initial_seed)

        return callbacks

    def _prepare_for_epoch(self, stage: str, epoch: int):
        return self.loaders

    def run_experiment(self, experiment: Experiment, check: bool = False):
        """
        Starts the experiment
        """
        self._check_run = check
        self.experiment = experiment

        # jupyter source code logging hack
        # + hack to prevent cycle imports
        from catalyst.dl.experiment import BaseExperiment
        if isinstance(self.experiment, BaseExperiment) \
                and self.experiment.logdir is not None:
            expdir = Path(os.getcwd())
            logdir = Path(self.experiment.logdir)
            utils.dump_base_experiment_code(expdir, logdir)

        try:
            for stage in self.experiment.stages:
                self._run_stage(stage)
        except (Exception, KeyboardInterrupt) as ex:
            # if an exception had been raised 
            # before the exception-handlers were initialized
            if self.loggers is None or self.callbacks is None:
                raise ex
            else:
                self.state.exception = ex
                self._run_event("exception", moment=None)

        return self


__all__ = ["DLRunner"]
