from typing import Callable  # isort:skip

from catalyst.core import Runner
from catalyst.dl import utils
from .experiment import DLExperiment
from .state import DLState


class DLRunner(Runner):
    experiment: DLExperiment
    state: DLState

    experiment_fn: Callable = DLExperiment
    state_fn: callable = DLState

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage=stage)

        # @TODO: remove this trick
        utils.set_global_seed(self.experiment.initial_seed)
        loaders = self.experiment.get_loaders(stage)
        self.loaders = loaders


__all__ = ["DLRunner"]
