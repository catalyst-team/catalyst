# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackScope
from catalyst.core.experiment import IExperiment
from catalyst.core.legacy import IRunnerLegacy
from catalyst.tools import settings
from catalyst.tools.frozen_class import FrozenClass
from catalyst.tools.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    Scheduler,
)


class RunnerException(Exception):
    """Exception class for all runner errors."""

    def __init__(self, message: str):
        """
        Args:
            message: exception message
        """
        super().__init__(message)


class IRunner(ABC, IRunnerLegacy, FrozenClass):
    """
    An abstraction that knows how to run an experiment.
    It contains all the logic of **how** to run the experiment,
    stages, epoch and batches.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment.IExperiment`
            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.dl.runner.runner.Runner`
        - :py:mod:`catalyst.dl.runner.supervised.SupervisedRunner`

    Runner also contains full information about experiment runner.


    Runner section


    **runner.model** - an instance of torch.nn.Module class, \
    (should implement ``forward`` method); \
    for example,
    ::

        runner.model = torch.nn.Linear(10, 10)

    **runner.device** - an instance of torch.device (CPU, GPU, TPU); \
    for example,
    ::

        runner.device = torch.device("cpu")


    Experiment section


    **runner.criterion** - an instance of torch.nn.Module class\
    or torch.nn.modules.loss._Loss (should implement ``forward`` method); \
    for example,
    ::

        runner.criterion = torch.nn.CrossEntropyLoss()

    **runner.optimizer** - an instance of torch.optim.optimizer.Optimizer\
    (should implement ``step`` method); \
    for example,
    ::

        runner.optimizer = torch.optim.Adam()

    **runner.scheduler** -
    an instance of torch.optim.lr_scheduler._LRScheduler\
    (should implement ``step`` method); \
    for example,
    ::

        runner.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()

    **runner.callbacks** -
    ordered dictionary with Catalyst.Callback instances;\
    for example,
    ::

        runner.callbacks = {
            "accuracy": AccuracyCallback(),
            "criterion": CriterionCallback(),
            "optim": OptimizerCallback(),
            "saver": CheckpointCallback()
        }


    Dataflow section


    **runner.loaders** - ordered dictionary with torch.DataLoaders; \
    for example,
    ::

        runner.loaders = {
            "train": MnistTrainLoader(),
            "valid": MnistValidLoader()
        }

    .. note::
        - "*train*" prefix is used for training loaders - \
          metrics computations, backward pass, optimization
        - "*valid*" prefix is used for validation loaders - \
          metrics computations only
        - "*infer*" prefix is used for inference loaders - \
          dataset prediction

    **runner.input** - dictionary, \
    containing batch of data from currents DataLoader; \
    for example,
    ::

        runner.input = {
            "images": np.ndarray(batch_size, c, h, w),
            "targets": np.ndarray(batch_size, 1),
        }

    **runner.output** - dictionary, \
    containing model output for current batch; \
    for example,
    ::

        runner.output = {"logits": torch.Tensor(batch_size, num_classes)}


    Metrics section


    **runner.batch_metrics** - dictionary, flatten storage for batch metrics; \
    for example,
    ::

        runner.batch_metrics = {"loss": ..., "accuracy": ..., "iou": ...}

    **runner.loader_metrics** - dictionary with aggregated batch statistics \
    for loader (mean over all batches) and global loader metrics, like AUC; \
    for example,
    ::

        runner.loader_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    **runner.epoch_metrics** - dictionary with summarized metrics \
    for different loaders and global epoch metrics, like lr, momentum; \
    for example,
    ::

        runner.epoch_metrics = {
            "train_loss": ..., "train_auc": ..., "valid_loss": ...,
            "lr": ..., "momentum": ...,
        }


    Validation metrics section


    **runner.main_metric** - string, containing name of metric of interest \
    for optimization, validation and checkpointing during training

    **runner.minimize_metric** - bool, indicator flag

        - ``True`` if we need to minimize metric during training,\
          like `Cross Entropy loss`
        - ``False`` if we need to maximize metric during training, \
          like `Accuracy` or `Intersection over Union`


    Validation section


    **runner.valid_loader** - string, name of validation loader \
    for metric selection, validation and model checkpoining

    **runner.valid_metrics** - dictionary with validation metrics\
    for currect epoch; \
    for example,
    ::

        runner.valid_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    .. note::
        subdictionary of epoch_metrics

    **runner.is_best_valid** - bool, indicator flag

        - ``True`` if this training epoch is best over all epochs
        - ``False`` if not

    **runner.best_valid_metrics** - dictionary with best validation metrics \
    during whole training process


    Distributed section


    **runner.distributed_rank** - distributed rank of current worker

    **runner.is_distributed_master** - bool, indicator flag

        - ``True`` if is master node (runner.distributed_rank == 0)
        - ``False`` if is worker node (runner.distributed_rank != 0)

    **runner.is_distributed_worker** - bool, indicator flag

        - ``True`` if is worker node (runner.distributed_rank > 0)
        - ``False`` if is master node (runner.distributed_rank <= 0)


    Experiment info section


    **runner.global_sample_step** - int, numerical indicator, counter for all\
    individual samples, that passes through our model during training,\
    validation and inference stages

    **runner.global_batch_step** - int, numerical indicator, counter for all
    batches, that passes through our model during training, validation and\
    inference stages

    **runner.global_epoch** - int, numerical indicator,
    counter for all epochs,\
    that have passed during model training, validation and\
    inference stages

    **runner.verbose** - bool, indicator flag

    **runner.is_check_run** - bool, indicator flag

        - ``True`` if you want to check you pipeline and \
          run only 2 batches per loader and 2 epochs per stage
        - ``False`` (default) if you want to just the pipeline

    **runner.need_early_stop** - bool, indicator flag \
    used for EarlyStopping and CheckRun Callbacks

        - ``True`` if we need to stop the training
        - ``False`` (default) otherwise

    **runner.need_exception_reraise** - bool, indicator flag

        - ``True`` (default) if you want to show exception \
          during pipeline and stop the training process
        - ``False`` otherwise


    Stage info section


    **runner.stage_name** - string, current stage name,\
    for example,
    ::

        runner.stage_name = "pretraining" / "training" / "finetuning" / etc

    **runner.num_epochs** - int, maximum number of epochs, \
    required for this stage

    **runner.is_infer_stage** - bool, indicator flag

        - ``True`` for inference stages
        - ``False`` otherwise


    Epoch info section


    **runner.epoch** - int, numerical indicator for current stage epoch


    Loader info section


    **runner.loader_sample_step** - int, numerical indicator \
    for number of samples passed through our model in current loader

    **runner.loader_batch_step** - int, numerical indicator \
    for batch index in current loader


    **runner.loader_name** - string, current loader name\
    for example,
    ::

        runner.loader_name = "train_dataset1" / "valid_data2" / "infer_golden"

    **runner.loader_len** - int, maximum number of batches in current loader

    **runner.loader_batch_size** - int, batch size parameter in current loader

    **runner.is_train_loader** - bool, indicator flag

        - ``True`` for training loaders
        - ``False`` otherwise

    **runner.is_valid_loader** - bool, indicator flag

        - ``True`` for validation loaders
        - ``False`` otherwise

    **runner.is_infer_loader** - bool, indicator flag

        - ``True`` for inference loaders
        - ``False`` otherwise


    Batch info section


    **runner.batch_size** - int, length of the current batch

    Logging section


    **runner.logdir** - string, path to logging directory to save\
    all logs, metrics, checkpoints and artifacts

    **runner.checkpoint_data** - dictionary\
    with all extra data for experiment tracking

    Extra section


    **runner.exception** - python Exception instance to raise (or not ;) )

    """

    _experiment_fn: Callable = IExperiment

    def __init__(
        self, model: RunnerModel = None, device: Device = None, **kwargs,
    ):
        """
        Args:
            model (RunnerModel): Torch model object
            device (Device): Torch device
        """
        self._device = None
        self._model = None
        self._prepare_inner_state(model=model, device=device)
        self._unfreeze()
        self._init(**kwargs)
        self._freeze()

    def _prepare_inner_state(
        self,
        stage: str = settings.stage_infer_prefix,
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
        verbose: bool = False,
        **kwargs,
    ):
        self._unfreeze()

        # main runner components: model and device to run
        self.device: Device = device
        self.model: RunnerModel = model

        # extra experiment components,
        # use `catalyst.core.IExperiment` to setup them
        self.criterion: RunnerCriterion = criterion
        self.optimizer: RunnerOptimizer = optimizer
        self.scheduler: RunnerScheduler = scheduler
        # and callbacks
        self.callbacks: Dict[str, "Callback"] = callbacks or {}

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

        # metrics & validation
        self.main_metric: str = main_metric
        self.minimize_metric: bool = minimize_metric

        # validation
        self.valid_loader: str = valid_loader
        self.valid_metrics: Dict = defaultdict(None)
        self.is_best_valid: bool = False
        self.best_valid_metrics: Dict = defaultdict(None)

        # distributed info
        self.distributed_rank: int = utils.get_rank()
        self.is_distributed_master: bool = ~(self.distributed_rank > 0)
        self.is_distributed_worker: bool = self.distributed_rank > 0
        # experiment info
        self.global_sample_step: int = 0
        self.global_batch_step: int = 0
        self.global_epoch: int = 1
        self.verbose: bool = verbose
        self.is_check_run: bool = is_check_run
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True
        # stage info
        self.num_epochs: int = num_epochs
        self.stage_name: str = stage
        self.is_infer_stage: bool = self.stage_name.startswith(
            settings.stage_infer_prefix
        )
        # epoch info
        self.epoch: int = 1
        # loader info
        self.loader_sample_step: int = 0
        self.loader_batch_step: int = 0
        self.loader_name: str = None
        self.loader_len: int = 0
        self.loader_batch_size = 0
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = True
        # batch info
        self.batch_size: int = 0

        # logging
        self.expdir: Path = None
        self.logdir: Path = Path(logdir) if logdir is not None else None
        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data: Dict = checkpoint_data or {}

        # extra
        self.exception: Optional[Exception] = None

        # kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._freeze()

    def _init(self, **kwargs) -> None:
        """
        Inner method for children's classes
        to specify type for Runners' Experiment.
        """
        self.experiment: IExperiment = None

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

        Raises:
            TypeError: if value is out of
                `torch.nn.Module` or `Dict[str, torch.nn.Module]`
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
        elif isinstance(value, type(None)):
            model = None
        else:
            raise TypeError(
                f"Invalid value type "
                f"must be `torch.nn.Module` or `Dict[str, torch.nn.Module]` "
                f"got '{type(value)}'"
            )

        if model is not None and self._device is not None:
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

        Raises:
            TypeError: if `value` is out of `torch.device`, `str` or `None`
        """
        if isinstance(value, torch.device):
            self._device = value
        elif isinstance(value, str):
            self._device = torch.device(value)
        elif isinstance(value, type(None)):
            self._device = None
        else:
            raise TypeError(
                f"Invalid value type "
                f"must be `str` or `torch.device` "
                f"got '{type(value)}'"
            )

        if self._model is not None:
            self._model = utils.maybe_recursive_call(
                self._model, "to", device=self._device or "cpu"
            )

    @staticmethod
    def _get_experiment_components(
        experiment: IExperiment, stage: str = None, device: Device = None,
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
        experiment: IExperiment, stage: str,
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

        Used to get a named attribute from a `IRunner` by `key` keyword;
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

        Returns:
            inner attribute
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
        Prepares callbacks with `self._get_experiment_callbacks`.
        Prepares inner state with `self._prepare_inner_state`

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
        """
        utils.set_global_seed(self.experiment.initial_seed)
        (
            model,
            criterion,
            optimizer,
            scheduler,
            device,
        ) = self._get_experiment_components(
            experiment=self.experiment, stage=stage, device=self.device
        )

        utils.set_global_seed(self.experiment.initial_seed)
        callbacks = self._get_experiment_callbacks(
            experiment=self.experiment, stage=stage
        )

        migrating_params = dict(**self.experiment.get_stage_params(stage))
        migrate_from_previous_stage = migrating_params.get(
            "migrate_from_previous_stage", True
        )
        if (
            migrate_from_previous_stage
            and getattr(self, "callbacks", None) is not None
        ):
            for key, value in self.callbacks.items():
                if value.scope == CallbackScope.experiment:
                    callbacks[key] = value

        callbacks = utils.sort_callbacks_by_order(callbacks)

        if migrate_from_previous_stage:
            migrating_params.update(
                {
                    "global_epoch": getattr(self, "global_epoch", 1),
                    "global_batch_step": getattr(self, "global_batch_step", 0),
                    "global_sample_step": getattr(
                        self, "global_sample_step", 0
                    ),
                    "resume": getattr(self, "resume", None),
                }
            )

        self._prepare_inner_state(
            stage=stage,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            **migrating_params,
        )

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
            self.batch_size = len(next(iter(batch.values())))
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
        if len(loader) == 0:
            raise RunnerException(
                f"DataLoader with name {self.loader_name} is empty."
            )

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

        for loader_name, loader in self.loaders.items():
            if len(loader) == 0:
                raise RunnerException(
                    f"DataLoader with name {loader_name} is empty."
                )

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

    def run_experiment(self, experiment: IExperiment = None) -> "IRunner":
        """
        Starts the experiment.

        Args:
            experiment (IExperiment): Experiment instance to use for Runner.

        Returns:
            self, `IRunner` instance after the experiment

        Raises:
            Exception: if during pipeline exception,
                no handler we found into callbacks
            KeyboardInterrupt: if during pipeline exception,
                no handler we found into callbacks
        """
        self.experiment = experiment or self.experiment
        assert self.experiment is not None

        try:
            for stage in self.experiment.stages:
                self._run_stage(stage)
        except (Exception, KeyboardInterrupt) as ex:
            from catalyst.core.callbacks.exception import ExceptionCallback

            def _exception_handler_check(callbacks: Union[OrderedDict, Dict]):
                return callbacks is not None and any(
                    issubclass(x.__class__, ExceptionCallback)
                    for x in callbacks.values()
                )

            if _exception_handler_check(getattr(self, "callbacks", None)):
                self.exception = ex
                self._run_event("on_exception")
            else:
                raise ex

        return self


class IStageBasedRunner(IRunner):
    """
    Runner abstraction that suppose to have constant
    datasources per stage.
    """

    def _prepare_for_stage(self, stage: str):
        """
        Inner method to prepare `Runner` for the specified stage.

        Sets `Experiment` initial seed.
        Prepares experiment components with `self._get_experiment_components`.
        Prepares callbacks with `self._get_experiment_callbacks`.
        Prepares inner state with `self._prepare_inner_state`
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


__all__ = ["IRunner", "IStageBasedRunner", "RunnerException"]
