from typing import Any, Dict, Iterable, Mapping, Tuple, Union  # isort:skip
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader, Dataset

from catalyst.utils.typing import Criterion, Model, Optimizer, Scheduler
from .callback import Callback


class Experiment(ABC):
    """
    Object containing all information required to run the experiment

    Abstract, look for implementations
    """

    @property
    @abstractmethod
    def initial_seed(self) -> int:
        """Experiment's initial seed value"""
        pass

    @property
    @abstractmethod
    def logdir(self) -> str:
        """Path to the directory where the experiment logs"""
        pass

    @property
    @abstractmethod
    def stages(self) -> Iterable[str]:
        """Experiment's stage names"""
        pass

    @property
    @abstractmethod
    def distributed_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 methond"""
        pass

    @property
    @abstractmethod
    def monitoring_params(self) -> Dict:
        """Dict with the parameters for monitoring services"""
        pass

    @abstractmethod
    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage"""
        pass

    @abstractmethod
    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage"""
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage"""
        pass

    @abstractmethod
    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        """Returns the optimizer for a given stage"""
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        """Returns the scheduler for a given stage"""
        pass

    def get_experiment_components(
        self, model: nn.Module, stage: str
    ) -> Tuple[Criterion, Optimizer, Scheduler]:
        """
        Returns the tuple containing criterion, optimizer and scheduler by
        giving model and stage.
        """
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return criterion, optimizer, scheduler

    @abstractmethod
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage"""
        pass

    def get_datasets(
        self,
        stage: str,
        **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        """Returns the datasets for a given stage and kwargs"""
        raise NotImplementedError

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage"""
        raise NotImplementedError

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        """Returns the data transforms for a given stage and mode"""
        raise NotImplementedError

    def get_native_batch(
        self,
        stage: str,
        loader: Union[str, int] = 0,
        data_index: int = 0
    ):
        """Returns a batch from experiment loader

        Args:
            stage (str): stage name
            loader (Union[str, int]): loader name or its index,
                default is the first loader
            data_index (int): index in dataset from the loader
        """
        loaders = self.get_loaders(stage)
        if isinstance(loader, str):
            _loader = loaders[loader]
        elif isinstance(loader, int):
            _loader = list(loaders.values())[loader]
        else:
            raise TypeError("Loader parameter must be a string or an integer")

        dataset = _loader.dataset
        collate_fn = _loader.collate_fn

        sample = collate_fn([dataset[data_index]])

        return sample


__all__ = ["Experiment"]
