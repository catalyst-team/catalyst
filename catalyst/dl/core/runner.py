from typing import Callable  # isort:skip

from catalyst.core import _Runner, State
from catalyst.dl import utils
from .experiment import Experiment


class Runner(_Runner):
    """
    Deep Learning Runner for different supervised, unsupervised, gan, etc runs
    """
    _experiment_fn: Callable = Experiment
    _state_fn: Callable = State

    def _init(self):
        self.experiment: Experiment = None
        self.state: State = None

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage=stage)

        utils.set_global_seed(self.experiment.initial_seed)
        loaders = self.experiment.get_loaders(stage=stage)
        self.state.loaders = loaders


__all__ = ["Runner"]
