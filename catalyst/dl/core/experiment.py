from abc import abstractmethod, ABC
from typing import Iterable, Mapping, Any, List, Tuple, Dict
from collections import OrderedDict

from torch import nn
from torch.utils.data import Dataset, DataLoader

from .callback import Callback
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class Experiment(ABC):
    """
    Object containing all information required to run the experiment

    Abstract, look for implementations
    """

    @property
    @abstractmethod
    def logdir(self) -> str:
        pass

    @property
    @abstractmethod
    def stages(self) -> Iterable[str]:
        pass

    @property
    @abstractmethod
    def distributed_params(self) -> Dict:
        pass

    @abstractmethod
    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def get_model(self, stage: str) -> _Model:
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_optimizer(
        self,
        stage: str,
        model: nn.Module
    ) -> _Optimizer:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        pass

    def get_experiment_components(
        self,
        model: nn.Module,
        stage: str
    ) -> Tuple[_Criterion, _Optimizer, _Scheduler]:
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return criterion, optimizer, scheduler

    @abstractmethod
    def get_callbacks(self, stage: str) -> "List[Callback]":
        pass

    def get_datasets(
        self,
        stage: str,
        **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        raise NotImplementedError

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        raise NotImplementedError

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        raise NotImplementedError


__all__ = ["Experiment"]
