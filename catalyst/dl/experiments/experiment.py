import torch
from collections import OrderedDict
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod, ABC
from typing import Iterable, Any, Mapping, Dict, List

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
    def get_total_epochs(self, stage: str):
        pass

    @abstractmethod
    def get_valid_loader(self, stage: str):
        pass

    @abstractmethod
    def get_main_metric(self, stage: str):
        pass

    @abstractmethod
    def get_minimize_metric(self, stage: str):
        pass

    @abstractmethod
    def get_model(self) -> _Model:
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_optimizer(self, stage: str, model) -> _Optimizer:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
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
            total_epochs=self.get_total_epochs(stage),
            valid_loader=self.get_valid_loader(stage),
            main_metric=self.get_main_metric(stage),
            minimize_metric=self.get_minimize_metric(stage)
        )

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        pass

    @abstractmethod
    def get_transforms(self, mode, stage: str = None):
        pass


class SimpleExperiment(Experiment):
    """
    Super-simple one-staged experiment you can use to declare experiment
    in code 
    """

    def __init__(
        self,
        model: _Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "List[Callback]",
        epochs: int = 1,
        logdir: str = None,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        criterion: _Criterion = None,
        optimizer: _Optimizer = None,
        scheduler: _Scheduler = None,
        transforms=None
    ):
        self._model = model
        self._loaders = loaders
        self._callbacks = callbacks
        self._transforms = transforms

        self._epochs = epochs
        self._logdir = logdir
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric

        self._scheduler = scheduler
        self._optimizer = optimizer
        self._criterion = criterion

    @property
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return ["base"]

    def get_total_epochs(self, stage: str):
        return self._epochs

    def get_valid_loader(self, stage: str):
        return self._valid_loader

    def get_main_metric(self, stage: str):
        return self._main_metric

    def get_minimize_metric(self, stage: str):
        return self._minimize_metric

    def get_model(self) -> _Model:
        return self._model

    def get_criterion(self, stage: str) -> _Criterion:
        return self._criterion

    def get_optimizer(self, stage: str, model=None) -> _Optimizer:
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "List[Callback]":
        return self._callbacks

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        return self._loaders

    def get_transforms(self, mode, stage: str = None):
        return self._transforms


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "epochs", "criterion_params", "optimizer_params", "scheduler_params",
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
        # TODO formatting from config keys by default
        return self._config["logdir"]

    @property
    def stages(self) -> Iterable[str]:
        stages_keys = self.stages_config.keys()
        return stages_keys

    def get_total_epochs(self, stage: str):
        return self.stages_config[stage]["epochs"]

    def get_valid_loader(self, stage: str):
        return self.stages_config[stage].get("valid_loader", "valid")

    def get_main_metric(self, stage: str):
        return self.stages_config[stage].get("main_metric", "loss")

    def get_minimize_metric(self, stage: str):
        return self.stages_config[stage].get("minimize_metric", True)

    def get_model(self) -> _Model:
        model = Registry.get_model(**self._config["model_params"])
        return model

    def get_criterion(self, stage: str) -> _Criterion:
        criterion_params = self.stages_config[stage].get("criterion_params", {})
        criterion = Registry.get_criterion(**criterion_params)
        return criterion

    def get_optimizer(self, stage: str, model) -> _Optimizer:
        fp16 = isinstance(model, Fp16Wrap)
        optimizer_params = self.stages_config[stage].get("optimizer_params", {})
        optimizer = Registry.get_optimizer(
            model, **optimizer_params, fp16=fp16
        )
        return optimizer

    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        scheduler_params = self.stages_config[stage].get("scheduler_params", {})
        scheduler = Registry.get_scheduler(optimizer, **scheduler_params)
        return scheduler

    @abstractmethod
    def get_datasets(self, **kwargs) -> "OrderedDict[str, Dataset]":
        pass

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_conf = dict(self.stages_config[stage]['data_params'])
        batch_size = data_conf.pop('batch_size')
        n_workers = data_conf.pop('n_workers')
        drop_last = data_conf.pop('drop_last', True)

        datasets = self.get_datasets(**data_conf)

        loaders = OrderedDict()
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size,
                shuffle=name.startswith('train'),
                num_workers=n_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last
            )

        return loaders

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks_params = self.stages_config[stage].get("callbacks_params", {})

        callbacks = OrderedDict()
        for key, value in callbacks_params.items():
            callback = Registry.get_callback(**value)
            callbacks[key] = callback

        return callbacks


__all__ = ["Experiment", "SimpleExperiment", "ConfigExperiment"]
