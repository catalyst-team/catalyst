from typing import Any, Dict, Mapping, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache, partial

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core.callback import Callback, CallbackScope, ICallback
from catalyst.core.engine import IEngine
from catalyst.core.experiment import IExperiment
from catalyst.core.functional import (
    filter_callbacks_by_node,
    sort_callbacks_by_order,
)
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.typing import (
    Device,
    Model,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
)
from catalyst.utils.loaders import validate_loaders
from catalyst.utils.misc import maybe_recursive_call, set_global_seed

BATCH_METRICS = Dict[str, float]
LOADER_METRICS = Dict[str, BATCH_METRICS]
EPOCH_METRICS = Dict[str, LOADER_METRICS]


@lru_cache(maxsize=42)
def _has_str_intersections(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


class RunnerException(Exception):
    """Exception class for all runner errors."""

    def __init__(self, message: str):
        """
        Args:
            message: exception message
        """
        super().__init__(message)


class IRunner(ICallback, ILogger, ABC):
    """
    An abstraction that knows **how** to run an experiment.
    contains all the logic of how to run the experiment,
    stages, epochs, loaders and batches.
    Runner also contains full information about experiment runner.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment.IExperiment`
            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.runners.runner.Runner`
        - :py:mod:`catalyst.runners.supervised.SupervisedRunner`


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
            "optimizer": OptimizerCallback(),
            "checkpointer": CheckpointCallback()
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

    **runner.batch** - dictionary, \
    containing batch of data from currents DataLoader \
    and model output for current batch; \
    for example,
    ::

        runner.input = {
            "images": torch.Tensor(batch_size, c, h, w),
            "targets": torch.Tensor(batch_size, 1),
            "logits": torch.Tensor(batch_size, num_classes)
        }


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


    **runner.stage** - string, current stage name,\
    for example,
    ::

        runner.stage = "pretraining" / "training" / "finetuning" / etc

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

    Extra section


    **runner.exception** - python Exception instance to raise (or not ;) )

    """

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        device: Device = None,
    ):
        """
        Args:
            model: Torch model object
            engine: IEngine instance
            device: Torch device object
        """
        # the core
        self._model: RunnerModel = model
        self._device = None
        self.engine: IEngine = engine
        self.experiment: IExperiment = None
        self.trial: ITrial = None
        # the data
        self.loaders: Dict[str, DataLoader] = None
        # the components
        self.criterion: RunnerCriterion = None
        self.optimizer: RunnerOptimizer = None
        self.scheduler: RunnerScheduler = None
        # the callbacks
        self.callbacks: Dict[str, Callback] = {}
        # the loggers
        self.loggers: Dict[str, ILogger] = {}

        # the dataflow - model input/output and other batch tensors
        self.batch: [Dict, torch.Tensor] = None

        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: BATCH_METRICS = defaultdict(None)
        self.loader_metrics: LOADER_METRICS = defaultdict(None)
        self.epoch_metrics: EPOCH_METRICS = defaultdict(None)
        # self.stage_metrics: Dict = defaultdict(None)
        # self.experiment_metrics: Dict = defaultdict(None)

        # experiment info
        self.experiment_key: str = None
        self.global_epoch_step: int = 0
        self.global_batch_step: int = 0
        self.global_sample_step: int = 0

        # stage info
        self.stage_key: str = "infer"
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        self.stage_epoch_len: int = 0
        self.stage_epoch_step: int = 0
        self.stage_batch_step: int = 0
        self.stage_sample_step: int = 0

        # loader info
        self.loader: DataLoader = None
        self.loader_key: str = None
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = True
        self.loader_batch_size: int = 0
        self.loader_batch_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0

        # batch info
        self.batch_size: int = 0

        # extra
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

    @property
    def model(self) -> Model:
        """Returns the runner's model instance."""
        return self._model

    @model.setter
    def model(self, value: Union[Model, Dict[str, Model]]):
        """
        Runner's model, useful for experiment tracing.

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
            model: Model = maybe_recursive_call(
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
            value: new torch device.

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
            self._model = maybe_recursive_call(
                self._model, "to", device=self._device or "cpu"
            )

    def log_metrics(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_metrics(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
                global_sample_step=self.global_sample_step,
                global_batch_step=self.global_batch_step,
                global_epoch_step=self.global_epoch_step,
                # stage info
                stage_key=self.stage_key,
                stage_epoch_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
                stage_batch_step=self.stage_batch_step,
                stage_sample_step=self.stage_sample_step,
                # loader info
                loader_key=self.loader_key,
                loader_batch_len=self.loader_batch_len,
                loader_batch_step=self.loader_batch_step,
                loader_sample_step=self.loader_sample_step,
            )

    def log_image(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_image(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
                global_sample_step=self.global_sample_step,
                global_batch_step=self.global_batch_step,
                global_epoch_step=self.global_epoch_step,
                # stage info
                stage_key=self.stage_key,
                stage_epoch_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
                stage_batch_step=self.stage_batch_step,
                stage_sample_step=self.stage_sample_step,
                # loader info
                loader_key=self.loader_key,
                loader_batch_len=self.loader_batch_len,
                loader_batch_step=self.loader_batch_step,
                loader_sample_step=self.loader_sample_step,
            )

    def log_hparams(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_hparams(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
            )

    def flush_log(self) -> None:
        for logger in self.loggers.values():
            logger.flush_log()

    def close_log(self) -> None:
        for logger in self.loggers.values():
            logger.close_log()

    def on_experiment_start(self, runner: "IRunner"):
        """Event handler for experiment start.

        Args:
            runner: IRunner instance.

        .. note::
            This event work only on IRunner.
        """
        assert self.experiment is not None
        self.experiment_key = self.experiment.name
        self.global_epoch_step: int = 0
        self.global_batch_step: int = 0
        self.global_sample_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

        self.trial = self.experiment.get_trial()
        self.engine = self.experiment.get_engine()
        self.loggers = self.experiment.get_loggers()
        self.log_hparams(hparams=self.experiment.hparams)

    def on_stage_start(self, runner: "IRunner"):
        """Event handler for stage start.

        Args:
            runner: IRunner instance.
        """
        assert self.stage_key is not None
        stage_params = self.experiment.get_stage_params(self.stage_key)
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        self.stage_epoch_len = stage_params["num_epochs"]
        self.stage_epoch_step: int = 0
        self.stage_batch_step: int = 0
        self.stage_sample_step: int = 0

    def on_epoch_start(self, runner: "IRunner"):
        """Event handler for epoch start.

        Args:
            runner: IRunner instance.

        Raises:
            RunnerException: if current DataLoader is empty.
        """
        self.global_epoch_step += 1
        self.stage_epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)

    def on_loader_start(self, runner: "IRunner"):
        """Event handler for loader start.

        Args:
            runner: IRunner instance.

        Raises:
            RunnerException: if current DataLoader is empty.
        """
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        self.loader_batch_size: int = 0
        self.loader_batch_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.epoch)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler for batch start.

        Args:
            runner: IRunner instance.
        """
        self.batch = self.engine.sync_device(tensor_or_module=self.batch)

        if isinstance(self.batch, dict):
            self.batch_size = len(next(iter(self.batch.values())))
        else:
            self.batch_size = len(self.batch[0])

        self.global_batch_step += 1
        self.stage_batch_step += 1
        self.loader_batch_step += 1
        self.global_sample_step += self.batch_size
        self.stage_sample_step += self.batch_size
        self.loader_sample_step += self.batch_size
        self.batch_metrics: Dict = defaultdict(None)

    def on_batch_end(self, runner: "IRunner"):
        """Event handler for batch end.

        Args:
            runner: IRunner instance.
        """
        self.log_metrics(metrics=self.batch_metrics, scope="batch")

    def on_loader_end(self, runner: "IRunner"):
        """Event handler for loader end.

        Args:
            runner: IRunner instance.
        """
        self.log_metrics(metrics=self.loader_metrics, scope="loader")
        self.epoch_metrics[self.loader_key] = self.loader_metrics.copy()

    def on_epoch_end(self, runner: "IRunner"):
        """Event handler for epoch end.

        Args:
            runner: IRunner instance.
        """
        self.log_metrics(metrics=self.epoch_metrics, scope="epoch")
        self.flush_log()

    def on_stage_end(self, runner: "IRunner"):
        """Event handler for stage end.

        Args:
            runner: IRunner instance.
        """
        self.engine.deinit_components()

    def on_experiment_end(self, runner: "IRunner"):
        """Event handler for experiment end.

        Args:
            runner: IRunner instance.

        .. note::
            This event work only on IRunner.
        """
        self.close_log()

    def on_exception(self, runner: "IRunner"):
        """Event handler for exception case.

        Args:
            runner: IRunner instance.

        Raises:
            exception: pipeline exception
        """
        raise self.exception

    def _run_event(self, event: str) -> None:
        """Inner method to run specified event on Runners' callbacks.

        Args:
            event(str): event name to run on callbacks.

        .. note::
            To learn more about Catalyst Callbacks mechanism, please follow
            :py:mod:`catalyst.core.callback.Callback` documentation.

        """
        if _has_str_intersections(event, ("_start",)):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if _has_str_intersections(event, ("_end", "_exception")):
            getattr(self, event)(self)

    @abstractmethod
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        pass

    def _run_batch(self) -> None:
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        self._run_event("on_loader_start")
        with torch.set_grad_enabled(self.is_train_loader):
            for self.loader_batch_step, self.batch in enumerate(self.loader):
                self._run_batch()
                if self.need_early_stop:
                    self.need_early_stop = False
                    break
        self._run_event("on_loader_end")

    def _run_epoch(self) -> None:
        self._run_event("on_epoch_start")
        for self.loader_key, self.loader in self.loaders.items():
            self._run_loader()
        self._run_event("on_epoch_end")

    def _run_stage(self) -> None:
        self._run_event("on_stage_start")
        while self.stage_epoch_step < self.stage_epoch_len:
            self._run_epoch()
            if self.need_early_stop:
                self.need_early_stop = False
                break
        self._run_event("on_stage_end")

    def _run_experiment(self) -> None:
        self._run_event("on_experiment_start")
        for self.stage_key in self.experiment.stages:
            if self.engine.rank < 0:
                # single-device branch (cpu, gpu, dp)
                self._run_stage()
            else:
                # ddp-device branch
                # mp.spawn(self._run_stage, num_process=self.engine.world_size)
                raise NotImplementedError()
        self._run_event("on_experiment_end")

    def run_experiment(self, experiment: IExperiment = None) -> "IRunner":
        """
        Starts the experiment.

        Args:
            experiment: Experiment instance to use for Runner.

        Returns:
            self, `IRunner` instance after the experiment
        """
        self.experiment = experiment or self.experiment
        try:
            self._run_experiment()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


class IStageBasedRunner(IRunner):
    """
    Runner abstraction that suppose to have constant
    datasources per stage.
    """

    def on_stage_start(self, runner: "IRunner") -> None:
        """Event handler for stage start.

        For the `IStageBasedRunner` case:

        - prepares loaders - our datasources
        - prepares model components - model, criterion, optimizer, scheduler
        - prepares callbacks for the current stage

        Args:
            runner: IRunner instance.
        """
        super().on_stage_start(runner)
        stage_params = self.experiment.get_stage_params(self.stage_key)
        set_global_seed(self.experiment.seed)

        loaders = self.experiment.get_loaders(stage=self.stage_key)
        loaders = validate_loaders(loaders)
        self.loaders = loaders

        migrate_model_from_previous_stage = stage_params.get(
            "migrate_model_from_previous_stage", True
        )
        # some custom logic is possible here
        if self.model is not None and migrate_model_from_previous_stage:
            model_fn = lambda: self.model
        else:
            model_fn = partial(self.experiment.get_model, stage=self.stage_key)

        # @TODO: we need a better approach here
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=model_fn,
            criterion_fn=partial(
                self.experiment.get_criterion, stage=self.stage_key
            ),
            optimizer_fn=partial(
                self.experiment.get_optimizer, stage=self.stage_key
            ),
            scheduler_fn=partial(
                self.experiment.get_scheduler, stage=self.stage_key
            ),
        )

        # @TODO: we could refactor here
        migrate_callbacks_from_previous_stage = stage_params.get(
            "migrate_callbacks_from_previous_stage", True
        )
        callbacks = self.experiment.get_callbacks(self.stage_key)
        callbacks = filter_callbacks_by_node(callbacks)
        callbacks = sort_callbacks_by_order(callbacks)
        if migrate_callbacks_from_previous_stage:
            for key, value in self.callbacks.items():
                if value.scope == CallbackScope.experiment:
                    callbacks[key] = value
        callbacks = sort_callbacks_by_order(callbacks)
        self.callbacks = callbacks

    def on_epoch_start(self, runner: "IRunner"):
        """Event handler for stage start.

        Args:
            runner: IRunner instance.

        Raises:
            RunnerException: if current DataLoader is empty.
        """
        super().on_epoch_start(runner)
        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise NotImplementedError(
                    f"DataLoader with name {loader_key} is empty."
                )

    def on_loader_start(self, runner: "IRunner"):
        """Event handler for loader start.

        Args:
            runner: IRunner instance.

        Raises:
            RunnerException: if current DataLoader is empty.
        """
        super().on_loader_start(runner)
        self.loader_batch_len = len(self.loader)
        if self.loader_batch_len == 0:
            raise NotImplementedError(
                f"DataLoader with name {self.loader_key} is empty."
            )


__all__ = ["IRunner", "IStageBasedRunner", "RunnerException"]
