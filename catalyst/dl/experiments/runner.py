from collections import OrderedDict
from typing import Mapping, Any
from abc import ABC, abstractmethod

import torch
from torch import nn

from catalyst.dl.callbacks import Callback
from catalyst.dl.metric_manager import TimerManager
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from . import Experiment


class Runner(ABC):

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
        self.timers = TimerManager()

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

    def _call_callbacks(self, event: str):

        if self.callbacks is not None:
            for callback in self.callbacks.values():
                getattr(callback, f"on_{event}")(self.state)

    def _handle_runner_metrics(self):
        values = {
            "base/lr": self.state.lr,
            "base/momentum": self.state.momentum,
            "loss": self.state.loss
        }

        values.update(self.timers.elapsed)

        values["base/samples_per_sec"] = \
            self.state.batch_size / self.timers.elapsed["base/batch_time"]

        self.state.metrics.add_batch_value(metrics_dict=values)

    def _run_loader(self, loader):
        self.timers.reset()

        self.timers.start("base/data_time")
        self.timers.start("base/batch_time")
        
        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self.timers.stop("base/data_time")
            self.timers.start("base/model_time")

            self.state.metrics.begin_batch()
            
            self._call_callbacks("batch_start")
            self._run_batch(batch)
            self.timers.stop("base/model_time")

            self.timers.stop("base/batch_time")
            

            self._call_callbacks("batch_end")

            self._handle_runner_metrics()
            self.state.metrics.end_batch()

            self.timers.reset()
            
            self.timers.start("base/batch_time")
            self.timers.start("base/data_time")

    def _run_epoch(self, loaders):
        for loader_name in loaders:
            self.state.loader_name = loader_name
            self.state.loader_len = len(loaders[loader_name])
            self.state.is_train = loader_name.startswith("train")

            self.model.train(self.state.is_train)
            
            self.state.metrics.begin_loader(self.state.loader_name)
            self._call_callbacks("loader_start")

            self._run_loader(loaders[loader_name])
            
            self.state.metrics.end_loader()
            self._call_callbacks("loader_end")

    def _run_stage(self, mode: str, stage: str):
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self._prepare_state(mode, stage)

        self._call_callbacks("stage_start")
        for epoch in range(self.state.total_epochs):
            self.state.epoch = epoch
            self.state.metrics.begin_epoch()
            self._call_callbacks("epoch_start")
            self._run_epoch(loaders)
            self.state.metrics.end_epoch()
            self._call_callbacks("epoch_end")
        self._call_callbacks("stage_end")

    def run(self, mode):
        for stage in self.experiment.stages:
            self._run_stage(mode, stage)

    @abstractmethod
    def batch_handler(self, batch: Mapping[str, Any]):
        pass



class SupervisedModelRunner(Runner):
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
