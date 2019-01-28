from collections import OrderedDict
from typing import Mapping, Any
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from . import Experiment


class ExperimentRunner(ABC):

    def __init__(self, experiment: Experiment):
        """

        :param experiment: Experiment to run
        :type experiment: Experiment
        """
        self.experiment = experiment
        self.model = self.experiment.model
        self.state: RunnerState = None
        self.stage = None
        self.device = None
        self.wrapped_model: nn.Module = None
        """Model object wrapped in fp16 or DataParallel"""

        self.loaders: OrderedDict[str, DataLoader] = None
        self.callbacks: OrderedDict[str, Callback] = None
        self.prepare_model()

    def prepare_model(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        self.wrapped_model, self.device = \
            UtilsFactory.prepare_model(self.model)

    def prepare_state(self, mode: str, stage: str):
        migrating_params = {}
        if self.state is not None:
            migrating_params.update({
                "step": self.state.step,
                "epoch": self.state.epoch + 1,
                "best_metrics": self.state.best_metrics
            })

        self.state = RunnerState(
            mode=mode,
            device=self.device,
            model=self.model,
            stage=self.stage,
            criterion=self.experiment.get_criterion(stage),
            optimizer=self.experiment.get_optimizer(stage),
            scheduler=self.experiment.get_scheduler(stage),
            **self.experiment.get_state_params(stage),
            **migrating_params
        )

    @abstractmethod
    def _run_batch(self, batch: Mapping[str, Any]):
        pass

    def _run_loader(self, loader_name: str):
        loader = self.loaders[loader_name]
        for i, batch in enumerate(loader):
            batch = self.move_batch_to_device(self.device, batch)
            self._handle_event("batch_start")
            self._run_batch(batch)
            self._handle_event("batch_end")

    def _run_epoch(self):
        for loader_name in self.loaders:
            self.state.loader_name = loader_name
            self.state.loader_len = len(self.loaders[loader_name])
            self.state.is_train = loader_name.startswith('train')

            self.wrapped_model.train(self.state.is_train)
            self._handle_event("loader_start")
            self._run_loader(loader_name)
            self._handle_event("loader_end")

    def _run_stage(self, mode: str, stage: str):
        self.loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self.prepare_state(mode, stage)

        for epoch in range(self.state.total_epochs):
            self.state.epoch = epoch
            self._handle_event("epoch_start")
            self._run_epoch()
            self._handle_event("epoch_end")

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

    def run(self, mode):
        self._handle_event("mode_start")
        for stage in self.experiment.stages:
            self._handle_event("stage_start")
            self._run_stage(mode, stage)
            self._handle_event("stage_end")
        self._handle_event("mode_end")

    @staticmethod
    def move_batch_to_device(device, batch: Mapping[str, Any]):
        res = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return res


class SupervisedModelRunner(ExperimentRunner):
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
        output = self.wrapped_model(batch[self.input_key])
        self.state.output = {self.output_key: output}
