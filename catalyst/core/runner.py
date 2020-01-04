from typing import Any, Mapping, Optional, Dict, Union  # isort:skip
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DistributedSampler, DataLoader

from catalyst.dl import utils
from catalyst.utils.typing import Device, Model
from .callback import Callback, LoggerCallback
from .state import State


class Runner(ABC):
    """
    Abstract class for all runners inherited from
    """
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

        self.state: State = None
        self.callbacks: OrderedDict[str, Callback] = None
        self.loggers: OrderedDict[str, LoggerCallback] = None

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
            values_are_models = all([
                isinstance(v, nn.Module) for v in value.values()
            ])
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
            model.to(device=self._device)

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
            self._model.to(device=self._device)

    @abstractmethod
    def _prepare_for_stage(self, stage: str):
        pass

    @abstractmethod
    def _prepare_for_epoch(self, stage: str, epoch: int):
        pass

    @abstractmethod
    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner

        Args:
            batch: Key-value batch items
            **kwargs: kwargs to pass to the model
        """
        pass

    def _batch2device(self, batch: Mapping[str, Any], device: Device):
        output = utils.any2device(batch, device)
        return output

    def _run_event(self, event: str, moment: Optional[str]):
        fn_name = f"on_{event}"
        if moment is not None:
            fn_name = f"{fn_name}_{moment}"

        # before callbacks
        if self.state is not None:
            getattr(self.state, f"{fn_name}_pre")()

        if self.loggers is not None and moment == "start":
            for logger in self.loggers.values():
                getattr(logger, fn_name)(self.state)

        # running callbacks
        if self.callbacks is not None:
            for callback in self.callbacks.values():
                getattr(callback, fn_name)(self.state)

        # after callbacks
        if self.loggers is not None and \
                (moment == "end" or moment is None):  # for on_exception case
            for logger in self.loggers.values():
                getattr(logger, fn_name)(self.state)

        if self.state is not None:
            getattr(self.state, f"{fn_name}_post")()

    def predict_batch(
        self,
        batch: Mapping[str, Any],
        **kwargs
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

    def _run_batch(self, batch):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.input = batch
        self.state.timer.stop("_timers/data_time")

        self._run_event("batch", moment="start")
        self.state.timer.start("_timers/model_time")
        self.state.output = self.forward(batch)
        self.state.timer.stop("_timers/model_time")
        self.state.timer.stop("_timers/batch_time")
        self._run_event("batch", moment="end")

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
            if self._check_run and i >= 3:
                break

            self.state.timer.start("_timers/batch_time")
            self.state.timer.start("_timers/data_time")

    def _run_epoch(self, stage: str, epoch: int):
        loaders = self._prepare_for_epoch(stage=stage, epoch=epoch)

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
                self.state.initial_seed + self.state.epoch + 1
            )
            self._run_event("loader", moment="start")
            with torch.set_grad_enabled(self.state.need_backward):
                self._run_loader(loader)
            self._run_event("loader", moment="end")

    def _run_stage(self, stage: str):
        callbacks = self._prepare_for_stage(stage)

        # loaders = self.experiment.get_loaders(stage)
        # callbacks = self.experiment.get_callbacks(stage)
        loggers = utils.process_callbacks(
            OrderedDict([
                (k, v) for k, v in callbacks.items()
                if isinstance(v, LoggerCallback)
            ])
        )
        callbacks = utils.process_callbacks(
            OrderedDict([
                (k, v) for k, v in callbacks.items()
                if not isinstance(v, LoggerCallback)
            ])
        )
        self.state.loggers = loggers
        self.loggers = loggers
        self.callbacks = callbacks

        self._run_event("stage", moment="start")
        for epoch in range(self.state.num_epochs):
            self.state.stage_epoch = epoch

            self._run_event("epoch", moment="start")
            self._run_epoch(stage=stage, epoch=epoch)
            self._run_event("epoch", moment="end")

            if self._check_run and self.state.epoch >= 3:
                break
            if self.state.early_stop:
                self.state.early_stop = False
                break

            self.state.epoch += 1
        self._run_event("stage", moment="end")


__all__ = ["Runner"]
