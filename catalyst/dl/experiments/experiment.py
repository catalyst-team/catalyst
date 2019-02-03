from collections import OrderedDict
from torch import nn, optim
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC
from typing import Iterable, Any, Mapping

from catalyst.dl.callbacks import Callback

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


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
    def model(self) -> _Model:
        pass

    @property
    @abstractmethod
    def stages(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_optimizer(self, stage: str) -> _Optimizer:
        pass

    @abstractmethod
    def get_total_epochs(self, stage: str):
        pass

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str) -> _Scheduler:
        pass

    @abstractmethod
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        pass

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        return dict(
            logdir=self.logdir,
            total_epochs=self.get_total_epochs(stage)
        )


class SimpleExperiment(Experiment):
    """
    Super-simple one-staged experiment you can use to declare experiment
    in code 
    """

    def __init__(
        self,
        logdir: str,
        model: _Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "OrderedDict[str, Callback]",
        epochs: int,
        criterion: _Criterion = None,
        optimizer: _Optimizer = None,
        scheduler: _Scheduler = None
    ):
        self._logdir = logdir
        self._callbacks = callbacks
        self._epochs = epochs
        self._loaders = loaders
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._criterion = criterion
        self._model = model

    @property
    def model(self) -> _Model:
        return self._model

    @property
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return ['default']

    def get_optimizer(self, stage: str) -> _Optimizer:
        return self._optimizer

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        return self._loaders

    def get_criterion(self, stage: str) -> _Criterion:
        return self._criterion

    def get_scheduler(self, stage: str) -> _Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        return self._callbacks

    def get_total_epochs(self, stage: str):
        return self._epochs


__all__ = ["Experiment", "SimpleExperiment"]
