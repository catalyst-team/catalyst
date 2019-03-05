import torch
from collections import OrderedDict
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod, ABC
from typing import Iterable, Any, Mapping, Dict, List

from catalyst.contrib.registry import Registry
from catalyst.dl.callbacks import Callback
from catalyst.dl.utils import UtilsFactory
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
    def get_model(self, stage: str) -> _Model:
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
    def get_callbacks(self, stage: str) -> "List[Callback]":
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
    def get_transforms(self, stage: str = None, mode: str = None):
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
        return ["train"]

    def get_total_epochs(self, stage: str):
        return self._epochs

    def get_valid_loader(self, stage: str):
        return self._valid_loader

    def get_main_metric(self, stage: str):
        return self._main_metric

    def get_minimize_metric(self, stage: str):
        return self._minimize_metric

    def get_model(self, stage: str) -> _Model:
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

    def get_transforms(self, stage: str = None, mode: str = None):
        return self._transforms


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "criterion_params", "optimizer_params", "scheduler_params",
        "data_params", "state_params", "callbacks_params"
    ]

    def __init__(self, config: Dict):
        self._config = config

        self.stages_config = self._prepare_stages_config(config["stages"])

    def _prepare_stages_config(self, stages_config):
        stages_defaults = {}
        for key in self.STAGE_KEYWORDS:
            stages_defaults[key] = stages_config.pop(key, {})
        for stage in stages_config:
            for key in self.STAGE_KEYWORDS:
                stages_config[stage][key] = merge_dicts(
                    stages_config[stage].get(key, {}),
                    stages_defaults.get(key, {})
                )
        return stages_config

    @property
    def logdir(self):
        # TODO formatting from config keys by default
        return self._config["args"]["logdir"]

    @property
    def stages(self) -> List[str]:
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    def get_total_epochs(self, stage: str):
        return self.stages_config[stage]["state_params"]["total_epochs"]

    def get_valid_loader(self, stage: str):
        return self.stages_config[stage]["state_params"]\
            .get("valid_loader", "valid")

    def get_main_metric(self, stage: str):
        return self.stages_config[stage]["state_params"]\
            .get("main_metric", "loss")

    def get_minimize_metric(self, stage: str):
        return self.stages_config[stage]["state_params"]\
            .get("minimize_metric", True)

    @abstractmethod
    def _prepare_model_for_stage(self, stage: str, model: _Model):
        pass

    def get_model(self, stage: str) -> _Model:
        model = Registry.get_model(**self._config["model_params"])
        stage_index = self.stages.index(stage)
        if stage_index > 0:
            checkpoint_path = \
                f"{self.logdir}/checkpoints/checkpoint.best.pth.tar"
            checkpoint = UtilsFactory.load_checkpoint(checkpoint_path)
            UtilsFactory.unpack_checkpoint(checkpoint, model=model)
        model = self._prepare_model_for_stage(stage, model)
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
    def get_datasets(self, stage: str, **kwargs) -> "OrderedDict[str, Dataset]":
        pass

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_conf = dict(self.stages_config[stage]["data_params"])
        batch_size = data_conf.pop("batch_size")
        n_workers = data_conf.pop("n_workers")
        drop_last = data_conf.pop("drop_last", True)

        datasets = self.get_datasets(stage=stage, **data_conf)

        loaders = OrderedDict()
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size,
                shuffle=name.startswith("train"),
                num_workers=n_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last
            )

        return loaders

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks_params = self.stages_config[stage].get("callbacks_params", {})

        callbacks = []
        for key, value in callbacks_params.items():
            callback = Registry.get_callback(**value)
            callbacks.append(callback)

        return callbacks


__all__ = ["Experiment", "SimpleExperiment", "ConfigExperiment"]
