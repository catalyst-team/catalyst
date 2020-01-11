from typing import Union  # isort:skip
from abc import abstractmethod

from catalyst.core import Experiment


class RLExperiment(Experiment):

    @property
    @abstractmethod
    def min_num_transitions(self) -> int:
        pass



__all__ = ["RLExperiment"]
