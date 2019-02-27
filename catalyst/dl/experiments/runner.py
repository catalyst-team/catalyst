from collections import OrderedDict
from typing import Mapping, Any, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn

from catalyst.dl.callbacks import Callback, \
    LossCallback, OptimizerCallback, SchedulerCallback, CheckpointCallback
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory
from . import Experiment, SimpleExperiment, ConfigExperiment


class Runner(ABC):
    _simple_exp_parser = SimpleExperiment
    _config_exp_parser = ConfigExperiment

    def __init__(
        self,
        model: nn.Module = None,
        config: Dict = None,
        device=None,
    ):
        """
        @TODO: write docs
        """
        assert model or config

        self.model: nn.Module = model
        self.device = device

        self.experiment: Experiment = self._config_exp_parser(config) \
            if config is not None \
            else None
        self._check_run = False
        self._verbose = False
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

    def _prepare_model(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """

        if self.model is None:
            self.model = self.experiment.get_model()

        self.model, self.device = \
            UtilsFactory.prepare_model(self.model)

    def _prepare_state(self, mode: str, stage: str):
        migrating_params = {}
        if self.state is not None:
            migrating_params.update({
                "step": self.state.step,
                "epoch": self.state.epoch + 1
            })

        self._prepare_model()
        criterion, optimizer, scheduler = \
            self.experiment.get_model_stuff(self.model, stage)

        self.state = RunnerState(
            mode=mode,
            stage=self.stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=self._verbose,
            **self.experiment.get_state_params(stage),
            **migrating_params
        )

    def _call_callbacks(self, event: str):

        if self.state is not None and hasattr(self.state, f"on_{event}_pre"):
            getattr(self.state, f"on_{event}_pre")()

        if self.callbacks is not None:
            for callback in self.callbacks.values():
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

        self.state.timer.start("base/batch_time")
        self.state.timer.start("base/data_time")

        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self.state.timer.stop("base/data_time")

            self._call_callbacks("batch_start")

            self.state.timer.start("base/model_time")
            self._run_batch(batch)
            self.state.timer.stop("base/model_time")

            self.state.timer.stop("base/batch_time")
            self._call_callbacks("batch_end")

            self.state.timer.reset()

            if self._check_run and i >= 3:
                break

            self.state.timer.start("base/batch_time")
            self.state.timer.start("base/data_time")

    def _run_epoch(self, loaders):
        assert self.state.valid_loader in loaders.keys(), \
            f"'{self.state.valid_loader}' " \
            f"should be in provided loaders: {list(loaders.keys())}"

        for loader_name in loaders:
            self.state.loader_name = loader_name
            self.state.loader_len = len(loaders[loader_name])
            self.state.is_train = loader_name.startswith("train")
            self.model.train(self.state.is_train)

            self._call_callbacks("loader_start")
            self._run_loader(loaders[loader_name])
            self._call_callbacks("loader_end")

    def _run_stage(self, mode: str, stage: str):
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self._prepare_state(mode, stage)
        self.state.stage = stage

        self._call_callbacks("stage_start")
        for epoch in range(self.state.total_epochs):
            self.state.epoch = epoch
            self.state.metrics.begin_epoch()
            self._call_callbacks("epoch_start")
            self._run_epoch(loaders)
            self.state.metrics.end_epoch()
            self._call_callbacks("epoch_end")
            if self._check_run and epoch >= 3:
                break
            if self.state._early_stop:
                break
        self._call_callbacks("stage_end")

    def run_experiment(self, mode, experiment, check_run=False):
        self.experiment = experiment
        self._check_run = check_run
        for stage in self.experiment.stages:
            self._run_stage(mode, stage)
        return self

    def _prepare_config_experiment(self, config):
        return self._config_exp_parser(config=config)

    def _prepare_simple_experiment(self, **kwargs):
        return self._simple_exp_parser(model=self.model, **kwargs)

    def _prepare_experiment(self, *, config, **kwargs):
        experiment = self._prepare_config_experiment(config) \
            if config is not None \
            else self._prepare_simple_experiment(**kwargs)
        return experiment

    def run(self, *, mode, config=None, **kwargs):
        self._verbose = kwargs.pop("verbose", False)
        check_run = kwargs.pop("check_run", False)
        experiment = self._prepare_experiment(config=config, **kwargs)
        return self.run_experiment(
            mode=mode,
            experiment=experiment,
            check_run=check_run)

    def train(self, *, config=None, **kwargs):
        return self.run(mode="train", config=config, **kwargs)

    def infer(self, *, config=None, **kwargs):
        return self.run(mode="infer", config=config, **kwargs)


class SupervisedRunner(Runner):
    """
    Runner for experiments with supervised model
    """

    def __init__(
        self,
        model: nn.Module = None,
        config: Dict = None,
        device=None,
        input_key: str = "features",
        output_key: str = "logits"
    ):
        """
        @TODO update docs

        :type output_key: str
        :type input_key: str

        :param input_key: Key in batch dict mapping to model input
        :param output_key: Key in output dict model output will be stored under
        """
        super().__init__(model=model, config=config, device=device)
        self.input_key = input_key
        self.output_key = output_key

    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch[self.input_key])
        output = {self.output_key: output}
        return output

    def _prepare_simple_experiment(self, **kwargs):
        callbacks: OrderedDict = kwargs.pop("callbacks")
        c_values = callbacks.values()
        default_callbacks = [
            ("criterion", LossCallback),
            ("optimizer", OptimizerCallback),
            ("scheduler", SchedulerCallback),
            ("_default_saver", CheckpointCallback),
        ]

        for key, value in default_callbacks:
            if (kwargs.get(key, None) or key.startswith("_default")) \
                    and not any(isinstance(x, value) for x in c_values):
                callbacks[f"_{key}"] = value()

        return self._simple_exp_parser(
            model=self.model,
            callbacks=callbacks,
            **kwargs)
