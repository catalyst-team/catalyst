from typing import Any, Callable, Dict, Mapping, Tuple, Union  # isort:skip

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core import utils
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)
from .callback import Callback, CallbackNode, CallbackScope
from .callbacks import ExceptionCallback
from .experiment import _Experiment
from .state import State


class _Runner(ABC):
    """
    Abstract class for all runners inherited from
    """
    _experiment_fn: Callable = _Experiment
    _state_fn: Callable = State

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
        self._model: Model = model
        self._device: Device = device
        self._init()

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

    def _init(self):
        self.experiment: _Experiment = None
        self.state: State = None

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

    def _get_state(
        self,
        stage: str,
        model: Model,
        criterion: Criterion,
        optimizer: Optimizer,
        scheduler: Scheduler,
        device: Device,
        callbacks: Dict[str, Callback],
    ):
        migrating_params = dict(**self.experiment.get_state_params(stage))
        migrate_from_previous_stage = \
            migrating_params.get("migrate_from_previous_stage", True)

        if migrate_from_previous_stage \
                and self.state is not None \
                and self.state.callbacks is not None:
            for key, value in self.state.callbacks.items():
                if value.scope == CallbackScope.Experiment:
                    callbacks[key] = value
            callbacks = utils.process_callbacks(callbacks)

        if self.state is not None and migrate_from_previous_stage:
            migrating_params.update(
                {
                    "global_step": self.state.global_step,
                    "global_epoch": self.state.global_epoch,
                    "resume": getattr(self.state, "resume", None),
                }
            )

        state = self._state_fn(
            stage=stage,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            **migrating_params
        )

        return state

    def _get_callbacks(self, stage: str):
        callbacks = self.experiment.get_callbacks(stage)

        # distributed run setting
        rank = utils.get_rank()
        if rank == 0:  # master node
            # remove worker-only callbacks on master node
            for k in list(
                filter(
                    lambda c: callbacks[c].node == CallbackNode.Worker,
                    callbacks
                )
            ):
                del callbacks[k]
        elif rank > 0:  # worker node
            # remove master-only callbacks on worker nodes
            for k in list(
                filter(
                    lambda c: callbacks[c].node == CallbackNode.Master,
                    callbacks
                )
            ):
                del callbacks[k]

        callbacks = utils.process_callbacks(callbacks)

        return callbacks

    def _prepare_for_stage(self, stage: str):
        utils.set_global_seed(self.experiment.initial_seed)
        self.model, criterion, optimizer, scheduler, self.device = \
            self._get_experiment_components(stage=stage)

        utils.set_global_seed(self.experiment.initial_seed)
        callbacks = self._get_callbacks(stage)

        utils.set_global_seed(self.experiment.initial_seed)
        self.state = self._get_state(
            stage=stage,
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            callbacks=callbacks,
        )

    def _prepare_for_epoch(self, stage: str, epoch: int):
        pass

    def _run_event(self, event: str):
        for callback in self.state.callbacks.values():
            getattr(callback, event)(self.state)

    def _batch2device(self, batch: Mapping[str, Any], device: Device):
        output = utils.any2device(batch, device)
        return output

    def _run_batch_train_step(self, batch: Mapping[str, Any]):
        self.state.batch_out = self.forward(batch)

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
        self.state.global_step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.batch_in = batch

        self._run_event("on_batch_start")
        self._run_batch_train_step(batch=batch)
        self._run_event("on_batch_end")

    def _run_loader(self, loader: DataLoader):
        self.state.batch_size = (
            loader.batch_sampler.batch_size
            if loader.batch_sampler is not None else loader.batch_size
        )
        self.state.global_step = (
            self.state.global_step
            or self.state.global_epoch * len(loader) * self.state.batch_size
        )

        for i, batch in enumerate(loader):
            self.state.loader_step = i + 1
            self._run_batch(batch)
            if self.state.need_early_stop:
                self.state.need_early_stop = False
                break

    def _run_epoch(self, stage: str, epoch: int):
        self._prepare_for_epoch(stage=stage, epoch=epoch)
        state: State = self.state

        assert state.loaders is not None
        loaders = state.loaders

        # @TODO: better solution with train/inference handling ?
        state.is_infer_stage = state.stage_name.startswith("infer")
        if not state.is_infer_stage:
            assert state.valid_loader in loaders.keys(), \
                f"'{state.valid_loader}' " \
                f"should be in provided loaders: {list(loaders.keys())}"
        else:
            # @TODO: add check for non distributed run for inference
            assert not any(x.startswith("train") for x in loaders.keys()), \
                "for inference no train loader should be passed"

        for loader_name, loader in loaders.items():
            state.loader_name = loader_name
            state.loader_len = len(loader)
            state.is_train_loader = loader_name.startswith("train")
            self.model.train(state.is_train_loader)

            if isinstance(loader.sampler, DistributedSampler) \
                    and not state.is_infer_stage:
                loader.sampler.set_epoch(state.epoch)

            utils.set_global_seed(
                self.experiment.initial_seed + state.global_epoch + 1
            )
            self._run_event("on_loader_start")
            with torch.set_grad_enabled(state.is_train_loader):
                self._run_loader(loader)
            self._run_event("on_loader_end")

    def _run_stage(self, stage: str):
        self._prepare_for_stage(stage)
        state: State = self.state

        self._run_event("on_stage_start")
        while state.epoch < state.num_epochs + 1:
            utils.set_global_seed(
                self.experiment.initial_seed + state.global_epoch + 1
            )
            self._run_event("on_epoch_start")
            self._run_epoch(stage=stage, epoch=state.epoch)
            self._run_event("on_epoch_end")

            if state.need_early_stop:
                state.need_early_stop = False
                break

            state.global_epoch += 1
            state.epoch += 1
        self._run_event("on_stage_end")

    def run_experiment(self, experiment: _Experiment):
        """
        Starts the experiment
        """
        self.experiment = experiment

        try:
            for stage in self.experiment.stages:
                self._run_stage(stage)
        except (Exception, KeyboardInterrupt) as ex:

            def _exception_handler_check(callbacks: OrderedDict):
                return (
                    callbacks is not None and any(
                        issubclass(x.__class__, ExceptionCallback)
                        for x in callbacks.values()
                    )
                )
            if self.state is not None and \
                    _exception_handler_check(self.state.callbacks):
                self.state.exception = ex
                self._run_event("on_exception")
            else:
                raise ex

        return self


__all__ = ["_Runner"]
