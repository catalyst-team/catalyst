from typing import Any, Dict, Iterable, Mapping
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from catalyst.core.callback import Callback, ICallback
from catalyst.core.engine import Engine, IEngine
from catalyst.core.experiment import IExperiment
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.typing import Criterion, Model, Optimizer, RunnerModel, Scheduler


class SingleStageExperiment(IExperiment):
    """Single-staged experiment to declare experiments in the code."""

    def __init__(
        self,
        model: RunnerModel,
        loaders: Dict[str, DataLoader],
        callbacks: Dict[str, Callback] = None,
        loggers: Dict[str, ILogger] = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        engine: IEngine = None,
        trial: ITrial = None,
        stage: str = "train",
        num_epochs: int = 1,
        hparams: Dict = None,
    ):
        self._loaders = loaders
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._callbacks = callbacks
        self._loggers = loggers
        self._engine = engine
        self._trial = trial

        self._stage = stage
        self._num_epochs = num_epochs
        self._hparams = hparams

    @property
    def name(self) -> str:
        return "experiment" if self._trial is None else f"experiment_{self._trial.number}"

    @property
    def hparams(self) -> Dict:
        if self._hparams is not None:
            return self._hparams
        if self._trial is not None:
            return self._trial.params
        return {}

    @property
    def stages(self) -> Iterable[str]:
        return [self._stage]

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        return {
            "num_epochs": self._num_epochs,
            "migrate_model_from_previous_stage": False,
            "migrate_callbacks_from_previous_stage": False,
        }

    def get_loaders(self, stage: str, epoch: int = None) -> "OrderedDict[str, DataLoader]":
        return self._loaders

    def get_model(self, stage: str) -> Model:
        return self._model

    def get_criterion(self, stage: str) -> Criterion:
        return self._criterion

    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "OrderedDict[str, ICallback]":
        return self._callbacks or {}

    def get_engine(self) -> IEngine:
        return self._engine or Engine()

    def get_trial(self) -> ITrial:
        return self._trial

    def get_loggers(self) -> Dict[str, ILogger]:
        return self._loggers or {}
