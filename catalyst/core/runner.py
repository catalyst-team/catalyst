from typing import Any, Callable, Dict, Mapping, Tuple, Union, Optional
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core import utils
from catalyst.tools import settings
from catalyst.tools.frozen_class import FrozenClass
from catalyst.tools.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    Scheduler,
)

from .callback import Callback, CallbackScope
from .callbacks import ExceptionCallback
from .experiment import _Experiment

RunnerModel = Union[Model, Dict[str, Model]]
RunnerCriterion = Union[Criterion, Dict[str, Criterion]]
RunnerOptimizer = Union[Optimizer, Dict[str, Optimizer]]
RunnerScheduler = Union[Scheduler, Dict[str, Scheduler]]


class _Runner(ABC, FrozenClass):
    """
    An abstraction that knows how to run an experiment.
    It contains all the logic of **how** to run the experiment,
    stages, epoch and batches.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment._Experiment`
            - :py:mod:`catalyst.core.runner._Runner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.dl.runner.runner.Runner`
        - :py:mod:`catalyst.dl.runner.supervised.SupervisedRunner`

    """

    _experiment_fn: Callable = _Experiment

    def __init__(
        self, model: RunnerModel = None, device: Device = None,
    ):
        """
        Args:
            model (StateModel): Torch model object
            device (Device): Torch device
        """
        self._prepare_inner_state(device=device,  model=model)
        self._init()
        self._freeze()

    def _prepare_inner_state(
        self,
        stage: str = settings.stage_infer_prefix,  # @TODO: wtf?
        device: Device = None,
        model: RunnerModel = None,
        criterion: RunnerCriterion = None,
        optimizer: RunnerOptimizer = None,
        scheduler: RunnerScheduler = None,
        callbacks: Dict[str, "Callback"] = None,
        logdir: str = None,
        num_epochs: int = 1,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = settings.loader_valid_prefix,
        checkpoint_data: Dict = None,
        is_check_run: bool = False,
        **kwargs,
    ):
        # main runner components: model and device to run
        self.device: Device = device
        self.model: RunnerModel = model

        # extra experiment components,
        # use `catalyst.core._Experiment` to setup them
        self.criterion: RunnerCriterion = criterion
        self.optimizer: RunnerOptimizer = optimizer
        self.scheduler: RunnerScheduler = scheduler
        # and callbacks
        self.callbacks: Dict[str, "Callback"] = callbacks

        # the data
        self.loaders: OrderedDict[str, DataLoader] = None
        # and the dataflow - model input, model output
        self.input = None
        self.output = None

        # metrics flow - batch, loader, epoch metrics
        # let's use flatten storage for batch metrics
        # batch_metrics = {'loss': ..., 'accuracy': ..., 'iou': ...}
        self.batch_metrics: Dict = defaultdict(None)
        # just aggregated (aka mean over all batches)
        # batch statistics for loader
        # and global loader metrics, like AUC
        # loader_metrics = {'loss': ..., 'accuracy': ..., `auc`: ...}
        self.loader_metrics: Dict = defaultdict(None)
        # summarized metrics for different loaders
        # and global epoch metrics, like lr, momentum
        # epoch_metrics = {
        # 'train_loss': ..., 'train_auc': ..., 'valid_loss': ...,
        # 'lr': ..., 'momentum': ...,
        # }
        self.epoch_metrics: Dict = defaultdict(None)

        # validation
        self.valid_loader: str = valid_loader
        self.valid_metrics: Dict = defaultdict(None)
        self.is_best_valid: bool = False
        self.best_valid_metrics: Dict = defaultdict(None)
        # metrics & validation
        self.main_metric: str = main_metric
        self.minimize_metric: bool = minimize_metric

        # distributed info
        self.distributed_rank: int = utils.get_rank()
        self.is_distributed_master: bool = ~(self.distributed_rank > 0)
        self.is_distributed_worker: bool = self.distributed_rank > 0
        # experiment info
        self.global_sample_step: int = 0
        self.global_batch_step: int = 0
        self.global_epoch: int = 1
        self.is_check_run: bool = is_check_run
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True
        # stage info
        self.stage_name: str = stage
        self.epoch: int = 1
        self.num_epochs: int = num_epochs
        self.is_infer_stage: bool = self.stage_name.startswith(
            settings.stage_infer_prefix
        )
        # loader info
        self.loader_name: str = None
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_len: int = 0
        self.loader_batch_size = 0
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = False
        # batch info
        self.batch_size: int = 0

        # logging
        self.logdir: Path = Path(logdir) if logdir is not None else None
        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data: Dict = checkpoint_data or {}

        # other
        self.exception: Optional[Exception] = None

        # kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _init(self) -> None:
        """
        Inner method for children's classes
        to specify types for Runners' Experiment and State.
        """
        self.experiment: _Experiment = None

    @property
    def model(self) -> Model:
        """Returns the runner's model instance."""
        return self._model

    @model.setter
    def model(self, value: Union[Model, Dict[str, Model]]):
        """
        Setter for the runner's model, useful for experiment tracing.

        Args:
            value (Union[Model, Dict[str, Model]]): new model.
        """
        if isinstance(value, nn.Module):
            model = value
        elif isinstance(value, dict):
            values_are_models = all(
                isinstance(v, nn.Module) for v in value.values()
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
        """Returns the runner's device instance."""
        return self._device

    @device.setter
    def device(self, value: Device):
        """
        Setter for the runner's device.

        Args:
            value (Device): new torch device.
        """
        if isinstance(value, torch.device):
            self._device = value
        elif isinstance(value, str):
            self._device = torch.device(value)
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

    @property
    def batch_in(self):
        """Alias for `state.input`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.batch_in` instead.
        """
        warnings.warn(
            "`state.batch_in` was deprecated, "
            "please use `state.input` instead",
            DeprecationWarning,
        )
        return self.input

    @property
    def batch_out(self):
        """Alias for `state.output`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.batch_out` instead.
        """
        warnings.warn(
            "`state.batch_out` was deprecated, "
            "please use `state.output` instead",
            DeprecationWarning,
        )
        return self.output

    @property
    def need_backward_pass(self):
        """Alias for `state.is_train_loader`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.is_train_loader` instead.
        """
        warnings.warn(
            "`need_backward_pass` was deprecated, "
            "please use `is_train_loader` instead",
            DeprecationWarning,
        )
        return self.is_train_loader

    @property
    def loader_step(self):
        """Alias for `state.loader_batch_step`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.loader_batch_step` instead.
        """
        warnings.warn(
            "`loader_step` was deprecated, "
            "please use `loader_batch_step` instead",
            DeprecationWarning,
        )
        return self.loader_batch_step

    @staticmethod
    def _get_experiment_components(
        experiment: _Experiment, stage: str = None, device: Device = None,
    ) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
        """
        Inner method for `Experiment` components preparation.

        Check available torch device, takes model from the experiment
        and creates stage-specified criterion, optimizer, scheduler for it.

        Args:
            stage (str): experiment stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:
            tuple: model, criterion, optimizer,
                scheduler and device for a given stage and model
        """
        (
            model,
            criterion,
            optimizer,
            scheduler,
        ) = experiment.get_experiment_components(stage)
        (
            model,
            criterion,
            optimizer,
            scheduler,
            device,
        ) = utils.process_components(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            distributed_params=experiment.distributed_params,
            device=device,
        )

        return model, criterion, optimizer, scheduler, device

    @staticmethod
    def _get_experiment_callbacks(
        experiment: _Experiment,
        stage: str,
    ) -> Dict[str, Callback]:
        """Inner method for `Callbacks` preparation.

        Takes callbacks from the Experiment
        and filters them for distributed master/worker cases.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary
                with callbacks for current experiment stage.
        """
        callbacks = experiment.get_callbacks(stage)
        callbacks = utils.filter_callbacks_by_node(callbacks)
        callbacks = utils.sort_callbacks_by_order(callbacks)
        return callbacks

    def get_attr(self, key: str, inner_key: str = None) -> Any:
        """
        Alias for python `getattr` method. Useful for Callbacks preparation
        and cases with multi-criterion, multi-optimizer setup.
        For example, when you would like to train multi-task classification.

        Used to get a named attribute from a `State` by `key` keyword;
        for example\
        ::

            # example 1
            runner.get_attr("criterion")
            # is equivalent to
            runner.criterion

            # example 2
            runner.get_attr("optimizer")
            # is equivalent to
            runner.optimizer

            # example 3
            runner.get_attr("scheduler")
            # is equivalent to
            runner.scheduler

        With `inner_key` usage, it suppose to find a dictionary under `key`\
        and would get `inner_key` from this dict; for example,
        ::

            # example 1
            runner.get_attr("criterion", "bce")
            # is equivalent to
            runner.criterion["bce"]

            # example 2
            runner.get_attr("optimizer", "adam")
            # is equivalent to
            runner.optimizer["adam"]

            # example 3
            runner.get_attr("scheduler", "adam")
            # is equivalent to
            runner.scheduler["adam"]

        Args:
            key (str): name for attribute of interest,
                like `criterion`, `optimizer`, `scheduler`
            inner_key (str): name of inner dictionary key
        """
        if inner_key is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[inner_key]

    def _prepare_for_stage(self, stage: str) -> None:
        """
        Inner method to prepare `Runner` for the specified stage.

        Sets `Experiment` initial seed.
        Prepares experiment components with `self._get_experiment_components`.
        Prepares callbacks with `self._get_callbacks`.
        Prepares `State` with `self._get_state`.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
        """
        utils.set_global_seed(self.experiment.initial_seed)
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.device,
        ) = self._get_experiment_components(
            experiment=self.experiment,
            stage=stage,
        )

        utils.set_global_seed(self.experiment.initial_seed)
        callbacks = self._get_experiment_callbacks(
            experiment=self.experiment,
            stage=stage
        )

        migrating_params = dict(**self.experiment.get_state_params(stage))
        migrate_from_previous_stage = migrating_params.get(
            "migrate_from_previous_stage", True
        )
        if migrate_from_previous_stage and self.callbacks is not None:
            for key, value in self.callbacks.items():
                if value.scope == CallbackScope.Experiment:
                    callbacks[key] = value
            callbacks = utils.sort_callbacks_by_order(callbacks)

        self.callbacks = callbacks

    def _prepare_for_epoch(self, stage: str, epoch: int) -> None:
        """
        Inner method to prepare `Runner` for the specified stage and epoch.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch (int): epoch index
        """
        pass

    def _run_event(self, event: str) -> None:
        """Inner method to run specified event on Runners' callbacks.

        Args:
            event(str): event name to run on callbacks.

        .. note::
            To learn more about Catalyst Callbacks mechanism, please follow
            :py:mod:`catalyst.core.callback.Callback` documentation.

        """
        for callback in self.callbacks.values():
            getattr(callback, event)(self)

    def _batch2device(
        self, batch: Mapping[str, Any], device: Device,
    ) -> Mapping[str, Any]:
        """
        Inner method to transfer incoming data batches to Runners' device.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            device (Device): torch device

        Returns:
            Mapping[str, Any]: same structure as value,
                but all tensors and np.arrays moved to device
        """
        output = utils.any2device(batch, device)
        return output

    @abstractmethod
    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        pass

    def _run_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to run train step on specified data batch,
        with batch callbacks events.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        if isinstance(batch, dict):
            self.batch_size = next(iter(batch.values())).shape[0]
        else:
            self.batch_size = len(batch[0])
        self.global_sample_step += self.batch_size
        self.loader_sample_step += self.batch_size
        batch = self._batch2device(batch, self.device)
        self.input = batch

        self._run_event("on_batch_start")
        self._handle_batch(batch=batch)
        self._run_event("on_batch_end")

    def _run_loader(self, loader: DataLoader) -> None:
        """
        Inner method to pass whole DataLoader through Runner,
        with loader callbacks events.

        Args:
            loader (DataLoader): dataloader to iterate
        """
        self.loader_batch_size = (
            loader.batch_sampler.batch_size
            if loader.batch_sampler is not None
            else loader.batch_size
        )

        self.loader_sample_step = 0
        for i, batch in enumerate(loader):
            self.global_batch_step += 1
            self.loader_batch_step = i + 1
            self._run_batch(batch)
            if self.need_early_stop:
                self.need_early_stop = False
                break

    def _run_epoch(self, stage: str, epoch: int) -> None:
        """
        Inner method to run epoch on Runner,
        with epoch callbacks events.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch (int): epoch index
        """
        self._prepare_for_epoch(stage=stage, epoch=epoch)
        assert self.loaders is not None

        # @TODO: better solution with train/inference handling ?
        self.is_infer_stage = self.stage_name.startswith("infer")
        if not self.is_infer_stage:
            assert self.valid_loader in self.loaders.keys(), (
                f"'{self.valid_loader}' "
                f"should be in provided loaders: {list(self.loaders.keys())}"
            )
        else:
            # @TODO: add check for non distributed run for inference
            assert not any(
                x.startswith(settings.loader_train_prefix)
                for x in self.loaders.keys()
            ), "for inference no train loader should be passed"

        for loader_name, loader in self.loaders.items():
            self.loader_name = loader_name
            self.loader_len = len(loader)
            self.is_train_loader = loader_name.startswith(
                settings.loader_train_prefix
            )
            self.is_valid_loader = loader_name.startswith(
                settings.loader_valid_prefix
            )
            self.is_infer_loader = loader_name.startswith(
                settings.loader_infer_prefix
            )
            utils.maybe_recursive_call(
                self.model, "train", mode=self.is_train_loader,
            )

            if (
                isinstance(loader.sampler, DistributedSampler)
                and not self.is_infer_stage
            ):
                loader.sampler.set_epoch(self.epoch)

            utils.set_global_seed(
                self.experiment.initial_seed + self.global_epoch + 1
            )
            self._run_event("on_loader_start")
            with torch.set_grad_enabled(self.is_train_loader):
                self._run_loader(loader)
            self._run_event("on_loader_end")

    def _run_stage(self, stage: str) -> None:
        """
        Inner method to run stage on Runner,
        with stage callbacks events.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        """
        self._prepare_for_stage(stage)

        self._run_event("on_stage_start")
        while self.epoch < self.num_epochs + 1:
            utils.set_global_seed(
                self.experiment.initial_seed + self.global_epoch + 1
            )
            self._run_event("on_epoch_start")
            self._run_epoch(stage=stage, epoch=self.epoch)
            self._run_event("on_epoch_end")

            if self.need_early_stop:
                self.need_early_stop = False
                break

            self.global_epoch += 1
            self.epoch += 1
        self._run_event("on_stage_end")

    def run_experiment(self, experiment: _Experiment = None) -> "_Runner":
        """
        Starts the experiment.

        Args:
            experiment (_Experiment): Experiment instance to use for Runner.

        """
        self.experiment = experiment or self.experiment
        assert self.experiment is not None

        try:
            for stage in self.experiment.stages:
                self._run_stage(stage)
        except (Exception, KeyboardInterrupt) as ex:

            def _exception_handler_check(callbacks: Union[OrderedDict, Dict]):
                return callbacks is not None and any(
                    issubclass(x.__class__, ExceptionCallback)
                    for x in callbacks.values()
                )

            if _exception_handler_check(self.callbacks):
                self.exception = ex
                self._run_event("on_exception")
            else:
                raise ex

        return self


class _StageBasedRunner(_Runner):
    """
    Runner abstraction that suppose to have constant
    datasources per stage.
    """

    def _prepare_for_stage(self, stage: str):
        """
        Inner method to prepare `Runner` for the specified stage.

        Sets `Experiment` initial seed.
        Prepares experiment components with `self._get_experiment_components`.
        Prepares callbacks with `self._get_callbacks`.
        Prepares `State` with `self._get_state`.
        Additionally sets `Experiment` datasources for specified stage.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
        """
        super()._prepare_for_stage(stage=stage)

        utils.set_global_seed(self.experiment.initial_seed)
        loaders = self.experiment.get_loaders(stage=stage)
        loaders = utils.validate_loaders(loaders)
        self.loaders = loaders


__all__ = ["_Runner", "_StageBasedRunner"]
