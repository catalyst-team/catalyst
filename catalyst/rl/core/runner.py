from typing import Dict, List

import numpy as np

import torch

from catalyst.core import Runner
from catalyst.rl import EnvironmentSpec
from .experiment import RLExperiment
from .state import RLRunnerState


class RLRunner(Runner):
    experiment: RLExperiment
    state: RLRunnerState

    def _prepare_for_epoch(self, stage: str, epoch: int):
        pass

    @torch.no_grad()
    def inference(
        self,
        sampler_ids: List[int],
        run_ids: List[int],
        states: np.ndarray,
        rewards: np.ndarray,
    ):
        # looks like production-ready thing
        # @TODO: make a microservice from this method
        raise NotImplementedError()

    def run(self):
        pass

    @classmethod
    def get_from_params(
        cls,
        algorithm_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        pass