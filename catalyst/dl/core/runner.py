from typing import Callable  # isort:skip

from catalyst.core import _Runner
from catalyst.dl import utils
from .experiment import Experiment
from .state import State


class Runner(_Runner):
    experiment: Experiment
    state: State

    experiment_fn: Callable = Experiment
    state_fn: callable = State

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage=stage)

        # @TODO: remove this trick
        utils.set_global_seed(self.experiment.initial_seed)
        loaders = self.experiment.get_loaders(stage=stage)
        self.loaders = loaders


__all__ = ["Runner"]
