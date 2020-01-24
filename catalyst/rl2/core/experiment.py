from typing import Union  # isort:skip
from abc import abstractmethod

from catalyst.core import Experiment
from catalyst.rl2 import AlgorithmSpec, EnvironmentSpec


class RLExperiment(Experiment):

    @property
    @abstractmethod
    def min_num_transitions(self) -> int:
        pass

    def get_algorithm(self, stage: str) -> AlgorithmSpec:
        """Returns the algorithm for a given stage"""
        pass

    def get_environment(self, stage: str) -> EnvironmentSpec:
        """Returns the environment for a given stage"""
        pass


__all__ = ["RLExperiment"]
