from typing import Dict, List, Iterable, Mapping, Any
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader

from catalyst.dl.core import Experiment, Callback
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class BaseExperiment(Experiment):
    """
    Super-simple one-staged experiment
        you can use to declare experiment in code
    """

    def __init__(
        self,
        model: _Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "List[Callback]" = None,
        logdir: str = None,
        stage: str = "train",
        criterion: _Criterion = None,
        optimizer: _Optimizer = None,
        scheduler: _Scheduler = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        distributed_params: Dict = None
    ):
        self._model = model
        self._loaders = loaders
        self._callbacks = callbacks or []

        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._logdir = logdir
        self._stage = stage
        self._num_epochs = num_epochs
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric
        self._verbose = verbose
        self._additional_state_kwargs = state_kwargs or {}
        self.checkpoint_data = checkpoint_data or {}
        self._distributed_params = distributed_params or {}

    @property
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return [self._stage]

    @property
    def distributed_params(self) -> Dict:
        return self._distributed_params

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        default_params = dict(
            logdir=self.logdir,
            num_epochs=self._num_epochs,
            valid_loader=self._valid_loader,
            main_metric=self._main_metric,
            verbose=self._verbose,
            minimize_metric=self._minimize_metric,
            checkpoint_data=self.checkpoint_data
        )
        state_params = {
            **self._additional_state_kwargs,
            **default_params
        }
        return state_params

    def get_model(self, stage: str) -> _Model:
        return self._model

    def get_criterion(self, stage: str) -> _Criterion:
        return self._criterion

    def get_optimizer(
        self,
        stage: str,
        model: nn.Module
    ) -> _Optimizer:

        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "List[Callback]":
        return self._callbacks

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        return self._loaders


__all__ = ["BaseExperiment"]
