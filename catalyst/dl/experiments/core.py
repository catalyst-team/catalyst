from abc import abstractmethod, ABC
from typing import Iterable, Mapping, Any, List, Tuple
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from ..callbacks.core import Callback

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
    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def get_model(self, stage: str) -> _Model:
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_optimizer_and_model(
        self,
        stage: str,
        model: nn.Module
    ) -> Tuple[_Optimizer, _Model]:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        pass

    def get_experiment_components(
        self,
        model: nn.Module,
        stage: str
    ) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler]:
        criterion = self.get_criterion(stage)
        optimizer, model = self.get_optimizer_and_model(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return model, criterion, optimizer, scheduler

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


class Runner(ABC):

    def __init__(
        self,
        model: nn.Module = None,
        device=None,
    ):
        """
        @TODO: write docs
        """
        # main
        self.model: nn.Module = model
        self.device = device
        self.experiment: Experiment = None
        self.state: RunnerState = None
        self.callbacks: List[Callback] = None

        # additional
        self._check_run = False

        if model is not None and device is None:
            self._prepare_model()

    def _batch2device(self, batch: Mapping[str, Any], device):
        res = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return res

    def _prepare_model(self, stage: str = None):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """

        if stage is not None:
            self.model = self.experiment.get_model(stage)

        self.model, self.device = \
            UtilsFactory.prepare_model(self.model)

    def _prepare_state(self, stage: str):
        migrating_params = {}
        if self.state is not None:
            migrating_params.update({
                "step": self.state.step,
                "epoch": self.state.epoch + 1
            })

        self._prepare_model(stage)
        self.model, criterion, optimizer, scheduler = \
            self.experiment.get_experiment_components(self.model, stage)

        self.state = RunnerState(
            stage=stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **self.experiment.get_state_params(stage),
            **migrating_params
        )

    def _run_event(self, event: str):

        if self.state is not None and hasattr(self.state, f"on_{event}_pre"):
            getattr(self.state, f"on_{event}_pre")()

        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, f"on_{event}")(self.state)

        if self.state is not None and hasattr(self.state, f"on_{event}_post"):
            getattr(self.state, f"on_{event}_post")()

    @abstractmethod
    def predict_batch(self, batch: Mapping[str, Any]):
        pass

    def _run_batch(self, batch):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.input = batch
        self.state.output = self.predict_batch(batch)

    def _run_loader(self, loader):
        self.state.batch_size = loader.batch_size
        self.state.step = (
            self.state.step
            or self.state.epoch * len(loader) * self.state.batch_size
        )
        # @TODO: remove time usage, use it under the hood
        self.state.timer.reset()

        self.state.timer.start("_timers/batch_time")
        self.state.timer.start("_timers/data_time")

        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self.state.timer.stop("_timers/data_time")

            self._run_event("batch_start")

            self.state.timer.start("_timers/model_time")
            self._run_batch(batch)
            self.state.timer.stop("_timers/model_time")

            self.state.timer.stop("_timers/batch_time")
            self._run_event("batch_end")

            self.state.timer.reset()

            if self._check_run and i >= 3:
                break

            self.state.timer.start("_timers/batch_time")
            self.state.timer.start("_timers/data_time")

    def _run_epoch(self, loaders):
        # @TODO: better solution with train/inference handling ?
        if not self.state.stage.startswith("infer"):
            assert self.state.valid_loader in loaders.keys(), \
                f"'{self.state.valid_loader}' " \
                f"should be in provided loaders: {list(loaders.keys())}"
        else:
            assert not any(x.startswith("train") for x in loaders.keys()), \
                "for inference no train loader should be passed"

        for loader_name in loaders:
            self.state.loader_name = loader_name
            self.state.loader_len = len(loaders[loader_name])
            self.state.need_backward = loader_name.startswith("train")
            self.model.train(self.state.need_backward)

            self._run_event("loader_start")
            with torch.set_grad_enabled(self.state.need_backward):
                self._run_loader(loaders[loader_name])
            self._run_event("loader_end")

    def _run_stage(self, stage: str):
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self._prepare_state(stage)
        self.state.stage = stage

        self._run_event("stage_start")
        for epoch in range(self.state.num_epochs):
            self.state.stage_epoch = epoch

            self._run_event("epoch_start")
            self._run_epoch(loaders)
            self._run_event("epoch_end")

            if self._check_run and self.state.epoch >= 3:
                break
            if self.state.early_stop:
                self.state.early_stop = False
                break

            self.state.epoch += 1
        self._run_event("stage_end")

    def run_experiment(
        self,
        experiment: Experiment,
        check: bool = False
    ):
        self._check_run = check

        self.experiment = experiment
        for stage in self.experiment.stages:
            self._run_stage(stage)
        return self
