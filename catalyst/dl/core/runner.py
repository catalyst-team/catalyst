from abc import abstractmethod, ABC
from typing import Tuple, List, Mapping, Any

import torch
from torch import nn
from torch.utils.data import DistributedSampler

from .callback import Callback
from .experiment import Experiment
from .state import RunnerState
from catalyst.dl import utils
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


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

    def _batch2device(self, batch: Mapping[str, Any], device):
        res = utils.any2device(batch, device)
        return res

    def _get_experiment_components(
        self,
        stage: str = None
    ) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """

        model = self.experiment.get_model(stage)
        criterion, optimizer, scheduler = \
            self.experiment.get_experiment_components(model, stage)

        model, criterion, optimizer, scheduler, device = \
            utils.process_components(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                distributed_params=self.experiment.distributed_params
            )

        return model, criterion, optimizer, scheduler, device

    def _prepare_state(self, stage: str):
        migrating_params = {}
        if self.state is not None:
            migrating_params.update({
                "step": self.state.step,
                "epoch": self.state.epoch + 1
            })

        self.model, criterion, optimizer, scheduler, self.device = \
            self._get_experiment_components(stage)

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
    def predict_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        pass

    def _run_batch(self, batch):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.input = batch
        self.state.timer.stop("_timers/data_time")

        self._run_event("batch_start")
        self.state.timer.start("_timers/model_time")
        self.state.output = self.predict_batch(batch)
        self.state.timer.stop("_timers/model_time")
        self.state.timer.stop("_timers/batch_time")
        self._run_event("batch_end")

    def _run_loader(self, loader):
        self.state.batch_size = (
            loader.batch_sampler.batch_size
            if loader.batch_sampler is not None
            else loader.batch_size)
        self.state.step = (
            self.state.step
            or self.state.epoch * len(loader) * self.state.batch_size
        )
        # @TODO: remove time usage, use it under the hood
        self.state.timer.reset()

        self.state.timer.start("_timers/batch_time")
        self.state.timer.start("_timers/data_time")

        for i, batch in enumerate(loader):
            self._run_batch(batch)

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

        for loader_name, loader in loaders.items():
            self.state.loader_name = loader_name
            self.state.loader_len = len(loader)
            self.state.need_backward = loader_name.startswith("train")
            self.model.train(self.state.need_backward)

            if isinstance(loader.sampler, DistributedSampler) \
                    and loader_name.startswith("train"):
                loader.sampler.set_epoch(self.state.stage_epoch)

            self._run_event("loader_start")
            with torch.set_grad_enabled(self.state.need_backward):
                self._run_loader(loader)
            self._run_event("loader_end")

    def _run_stage(self, stage: str):
        self._prepare_state(stage)
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

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


__all__ = ["Runner"]
