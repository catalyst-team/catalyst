from collections import OrderedDict
from torch import nn, optim
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC
from typing import Iterable, Any, Mapping, Dict

from catalyst.contrib.registry import Registry
from catalyst.dl.callbacks import Callback
from catalyst.dl.fp16 import Fp16Wrap
from catalyst.utils.misc import merge_dicts

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
    def stages(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        pass

    @abstractmethod
    def get_total_epochs(self, stage: str):
        pass

    @property
    @abstractmethod
    def model(self) -> _Model:
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_optimizer(self, stage: str, model=None) -> _Optimizer:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        pass

    def get_model_stuff(self, model, stage: str):
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return criterion, optimizer, scheduler

    @abstractmethod
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        pass

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        return dict(
            logdir=self.logdir,
            total_epochs=self.get_total_epochs(stage)
        )


class BaseExperiment(Experiment):
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
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return ["base"]

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        return self._loaders

    def get_total_epochs(self, stage: str):
        return self._epochs

    @property
    def model(self) -> _Model:
        return self._model

    def get_criterion(self, stage: str) -> _Criterion:
        return self._criterion

    def get_optimizer(self, stage: str, model=None) -> _Optimizer:
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        return self._callbacks


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "criterion_params", "optimizer_params", "scheduler_params",
        "stage_params", "state_params", "data_params", "callbacks_params"
    ]

    def __init__(self, config: Dict):
        self._config = config

        self.stages_config = self._prepare_stages_config(config["stages"])

    def _prepare_stages_config(self, stages_config):
        stages_defaults = {}
        for key in self.STAGE_KEYWORDS:
            stages_defaults[key] = stages_config.pop(key, {})

        for stages in stages_config:
            for key in self.STAGE_KEYWORDS:
                stages_config[stages][key] = merge_dicts(
                    stages_config[stages][key], stages_defaults.get(key, {})
                )

        return stages_config

    @property
    def logdir(self):
        return self._config["args"]["logdir"]

    @property
    def stages(self) -> Iterable[str]:
        stages_keys = self.stages_config.keys()
        return stages_keys

    def get_total_epochs(self, stage: str):
        return self.stages_config[stage]["args"]["epochs"]

    def get_model(self) -> _Model:
        model = Registry.get_model(**self._config["model_params"])
        return model

    def get_criterion(self, stage: str) -> _Criterion:
        criterion_params = self.stages_config[stage].get("criterion_params", {})
        criterion = Registry.get_criterion(**criterion_params)
        return criterion

    def get_optimizer(self, stage: str, model=None) -> _Optimizer:
        fp16 = isinstance(model, Fp16Wrap)
        optimizer_params = self.stages_config[stage].get("optimizer_params", {})
        optimizer = Registry.get_optimizer(
            model, **optimizer_params, fp16=fp16
        )
        return optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        scheduler_params = self.stages_config[stage].get("scheduler_params", {})
        scheduler = Registry.get_scheduler(optimizer, **scheduler_params)
        return scheduler

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks_params = self.stages_config[stage].get("callbacks_params", {})

        callbacks = OrderedDict()
        for key, value in callbacks_params.items():
            callback = Registry.get_callback(**value)
            callbacks[key] = callback

        return callbacks

    def get_transforms(self, stage: str = None, **kwargs):
        assert len(kwargs) == 0
        raise NotImplementedError

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        raise NotImplementedError


__all__ = ["Experiment", "BaseExperiment", "ConfigExperiment"]
