from collections import OrderedDict
from typing import Mapping, Any
from abc import ABC, abstractmethod

import torch
from torch import nn

from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from . import Experiment


class ExperimentModelRunner(ABC):

    def __init__(
            self,
            experiment: Experiment=None,
            model: nn.Module=None,
            device=None):
        """

        :param experiment: Experiment to run
        :type experiment: Experiment
        """
        assert experiment or model

        self.experiment = experiment
        self.model: nn.Module = model
        self.device = device

        self.state: RunnerState = None
        self.stage: str = None

        self.callbacks: OrderedDict[str, Callback] = None

        if device is None:
            self._prepare_model()

    @staticmethod
    def _batch2device(batch: Mapping[str, Any], device):
        res = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return res

    @abstractmethod
    def _run_batch(self, batch: Mapping[str, Any]):
        pass

    def _prepare_model(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """

        if self.model is None:
            self.model = self.experiment.model

        self.model, self.device = \
            UtilsFactory.prepare_model(self.model)

    def _prepare_state(self, mode: str, stage: str):
        migrating_params = {}
        if self.state is not None:
            migrating_params.update({
                "step": self.state.step,
                "epoch": self.state.epoch + 1,
                "best_metrics": self.state.best_metrics
            })

        self._prepare_model()
        criterion, optimizer, scheduler = self.experiment.get_model_stuff(
            self.model, stage)

        self.state = RunnerState(
            mode=mode,
            stage=self.stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **self.experiment.get_state_params(stage),
            **migrating_params
        )

    def _handle_event(self, event: str):
        pre_event_name = f"on_{event}_pre"
        post_event_name = f"on_{event}_post"

        if self.state is not None and hasattr(self.state, pre_event_name):
            getattr(self.state, pre_event_name)()

        if self.callbacks is not None:
            for callback in self.callbacks.values():
                getattr(callback, f"on_{event}")(self.state)

        if self.state is not None and hasattr(self.state, post_event_name):
            getattr(self.state, post_event_name)()

    def _run_loader(self, loader):
        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self._handle_event("batch_start")
            self._run_batch(batch)
            self._handle_event("batch_end")

    def _run_epoch(self, loaders):
        for loader_name in loaders:
            self.state.loader_name = loader_name
            self.state.loader_len = len(loaders[loader_name])
            self.state.is_train = loader_name.startswith("train")

            self.model.train(self.state.is_train)
            self._handle_event("loader_start")
            self._run_loader(loaders[loader_name])
            self._handle_event("loader_end")

    def _run_stage(self, mode: str, stage: str):
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self._prepare_state(mode, stage)

        self._handle_event("stage_start")
        for epoch in range(self.state.total_epochs):
            self.state.epoch = epoch
            self._handle_event("epoch_start")
            self._run_epoch(loaders)
            self._handle_event("epoch_end")
        self._handle_event("stage_end")

    def run(self, mode):
        self._handle_event("mode_start")  # @TODO: does not work
        for stage in self.experiment.stages:
            self._run_stage(mode, stage)
        self._handle_event("mode_end")   # @TODO: does not work

    @abstractmethod
    def batch_handler(self, batch: Mapping[str, Any]):
        pass


class SupervisedModelRunner(ExperimentModelRunner):
    """
    Runner for experiments with supervised model
    """

    def __init__(
        self,
        experiment: Experiment,
        input_key: str = "features",
        output_key: str = "logits"
    ):
        """
        :type experiment: Experiment
        :type output_key: str
        :type input_key: str

        :param experiment: Experiment to run
        :param input_key: Key in batch dict mapping to model input
        :param output_key: Key in output dict model output will be stored under
        """
        super().__init__(experiment)
        self.input_key = input_key
        self.output_key = output_key

    def _run_batch(self, batch: Mapping[str, Any]):
        self.state.input = batch
        output = self.batch_handler(batch)
        self.state.output = {self.output_key: output}

    def batch_handler(self, batch: Mapping[str, Any]):
        return self.model(batch[self.input_key])
