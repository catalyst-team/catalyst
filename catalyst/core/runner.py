from typing import (  # isort:skip
    Any, Callable, Dict, Mapping, Optional, Tuple, Union  # isort:skip
)  # isort:skip

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from catalyst import utils
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)
from .callback import Callback, LoggerCallback
from .experiment import _Experiment
from .state import _State


class _Runner(ABC):
    """
    Abstract class for all runners inherited from
    """

    experiment_fn: Callable = _Experiment
    state_fn: callable = _State

    def __init__(
        self,
        model: Model = None,
        device: Device = None,
    ):
        """
        Args:
            model (Model): Torch model object
            device (Device): Torch device
        """
        # main
        self._model: Model = model
        self._device: Device = device

        self.experiment: _Experiment = None
        self.state: _State = None

        self.callbacks: OrderedDict[str, Callback] = None
        self.loggers: OrderedDict[str, LoggerCallback] = None
        self.loaders: OrderedDict[str, DataLoader] = None

        # additional
        self._check_run = False

    @property
    def model(self) -> Model:
        """
        Returns the runner's model instance
        """
        return self._model

    @model.setter
    def model(self, value: Union[Model, Dict[str, Model]]):
        """
        Setter for the runner's model'
        """
        if isinstance(value, nn.Module):
            model = value
        elif isinstance(value, dict):
            values_are_models = all(
                [isinstance(v, nn.Module) for v in value.values()]
            )
            if not values_are_models:
                raise TypeError(
                    "Invalid dict value type, must be `torch.nn.Module`"
                )

            model = value

        else:
            raise TypeError(
                f"Invalid value type "
                f"must be `torch.nn.Module` or `Dict[str, torch.nn.Module]` "
                f"got '{type(value)}'"
            )

        if self._device is not None:
            model: Model = utils.maybe_recursive_call(
                model, "to", device=self._device
            )

        self._model = model

    @property
    def device(self) -> Device:
        """
        Returns the runner's device instance
        """
        return self._device

    @device.setter
    def device(self, value: Device):
        """
        Setter for the runner's device'
        """
        if isinstance(value, (str, torch.device)):
            self._device = value
        else:
            raise TypeError(
                f"Invalid value type "
                f"must be `str` or `torch.device` "
                f"got '{type(value)}'"
            )

        if self._model is not None:
            self._model = utils.maybe_recursive_call(
                self._model, "to", device=self._device
            )

    @abstractmethod
    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner

        Args:
            batch: Key-value batch items
            **kwargs: kwargs to pass to the model
        """
        pass

    def _get_experiment_components(
        self, stage: str = None
    ) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        utils.set_global_seed(self.experiment.initial_seed)
        model = self.experiment.get_model(stage)
        criterion, optimizer, scheduler = \
            self.experiment.get_experiment_components(model, stage)

        model, criterion, optimizer, scheduler, device = \
            utils.process_components(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                distributed_params=self.experiment.distributed_params,
                device=self.device
            )

        return model, criterion, optimizer, scheduler, device

    def _process_callbacks_for_stage(self, callbacks:):
        loggers = utils.process_callbacks(
            OrderedDict(
                [
                    (k, v) for k, v in callbacks.items()
                    if isinstance(v, LoggerCallback)
                ]
            )
        )
        callbacks = utils.process_callbacks(
            OrderedDict(
                [
                    (k, v) for k, v in callbacks.items()
                    if not isinstance(v, LoggerCallback)
                ]
            )
        )
        self.state.loggers = loggers
        self.loggers = loggers
        self.callbacks = callbacks
        # Prepare events dictionary
        start_callbacks = OrderedDict(**loggers, **callbacks)
        end_callbacks = OrderedDict(**callbacks, **loggers)
        start_events = ["stage_start", "epoch_start", "batch_start", "loader_start"]
        end_events = ["stage_end", "epoch_end", "batch_end", "loader_end", "exception"]
        stage_callbacks = {}
        for event in start_events:
            stage_callbacks[event] = start_callbacks
        for event in end_events:
            stage_callbacks[event] = end_callbacks
        return stage_callbacks


    def _prepare_for_stage(self, stage: str):
        utils.set_global_seed(self.experiment.initial_seed)

        migrating_params = dict(**self.experiment.get_state_params(stage))
        migrate_from_previous_stage = \
            migrating_params.get("migrate_from_previous_stage", True)
        if self.state is not None and migrate_from_previous_stage:
            migrating_params.update(
                {
                    "step": self.state.step,
                    "epoch": self.state.epoch,
                    "resume": getattr(self.state, "resume", None),
                }
            )

        utils.set_global_seed(self.experiment.initial_seed)
        self.model, criterion, optimizer, scheduler, self.device = \
            self._get_experiment_components(stage)

        utils.set_global_seed(self.experiment.initial_seed)
        self.state = self.state_fn(
            stage=stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **migrating_params
        )

        utils.set_global_seed(self.experiment.initial_seed)
        callbacks = self.experiment.get_callbacks(stage)

        self.stage_callbacks = self._process_callbacks_for_stage(callbacks)

    def _prepare_for_epoch(self, stage: str, epoch: int):
        pass

    def _run_event(self, event: str):
        fn_name = f"on_{event}"
        state_pre_fn_name = f"{fn_name}_pre"
        state_post_fn_name = f"{fn_name}_post"
        getattr(self.state, state_pre_fn_name)()
        for callback in self.stage_callbacks[event].values():
            getattr(callback, fn_name)(self.state)
        getattr(self.state, state_post_fn_name)()

    def _batch2device(self, batch: Mapping[str, Any], device: Device):
        output = utils.any2device(batch, device)
        return output

    def _run_batch_train_step(self, batch: Mapping[str, Any]):
        self.state.output = self.forward(batch)

    @torch.no_grad()
    def predict_batch(
        self, batch: Mapping[str, Any], **kwargs
    ) -> Mapping[str, Any]:
        """
        Run model for a batch of elements
        WARN: You should not override this method. If you need specific model
        call, override forward() method
        Args:
            batch: Key-value batch items
            **kwargs: kwargs to pass to the model

        Returns:
            model output key-value
        """
        batch = self._batch2device(batch, self.device)
        output = self.forward(batch, **kwargs)
        return output

    def _run_batch(self, batch: Mapping[str, Any]):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.input = batch
        self.state.timer.stop("_timers/data_time")

        self._run_event("batch_start")

        self.state.timer.start("_timers/model_time")
        self._run_batch_train_step(batch=batch)
        self.state.timer.stop("_timers/model_time")

        self.state.timer.stop("_timers/batch_time")
        self._run_event("batch_end")

    def _run_loader(self, loader: DataLoader):
        self.state.batch_size = (
            loader.batch_sampler.batch_size
            if loader.batch_sampler is not None else loader.batch_size
        )
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
            if self._check_run and i >= 2:
                break

            self.state.timer.start("_timers/batch_time")
            self.state.timer.start("_timers/data_time")

    def _run_epoch(self, stage: str, epoch: int):
        self._prepare_for_epoch(stage=stage, epoch=epoch)

        assert self.loaders is not None
        loaders = self.loaders

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

            utils.set_global_seed(
                self.experiment.initial_seed + self.state.epoch + 1
            )
            self._run_event("loader_start")
            with torch.set_grad_enabled(self.state.need_backward):
                self._run_loader(loader)
            self._run_event("loader_end")

    def _run_stage(self, stage: str):
        self._prepare_for_stage(stage)

        # checkpoint loading
        self._run_event("stage_start")

        while self.state.stage_epoch < self.state.num_epochs:
            self._run_event("epoch_start")
            utils.set_global_seed(
                self.experiment.initial_seed + self.state.epoch + 1
            )
            self._run_epoch(stage=stage, epoch=self.state.stage_epoch)
            self._run_event("epoch_end")

            if self._check_run and self.state.stage_epoch >= 2:
                break
            if self.state.early_stop:
                self.state.early_stop = False
                break

            self.state.epoch += 1
            self.state.stage_epoch += 1
        self._run_event("stage_end")

    def run_experiment(self, experiment: _Experiment, check: bool = False):
        """
        Starts the experiment
        """
        self._check_run = check
        self.experiment = experiment

        # jupyter source code logging hack
        # + hack to prevent cycle imports
        # @TODO: remove hack to catalyst.dl only, not core
        # from catalyst.dl.experiment import BaseExperiment
        # if isinstance(self.experiment, BaseExperiment) \
        #         and self.experiment.logdir is not None:
        #     expdir = Path(os.getcwd())
        #     logdir = Path(self.experiment.logdir)
        #     utils.dump_base_experiment_code(expdir, logdir)

        try:
            for stage in self.experiment.stages:
                self._run_stage(stage)
        except (Exception, KeyboardInterrupt) as ex:
            # if an exception had been raised
            # before the exception-handlers were initialized
            if self.loggers is None or self.callbacks is None:
                raise ex
            else:
                self.state.exception = ex
                self._run_event("exception")

        return self


__all__ = ["_Runner"]
