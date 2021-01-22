from typing import Any, Dict, Mapping, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache, partial
import logging

import torch
from torch import nn
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core.callback import Callback, CallbackScope, ICallback
from catalyst.core.engine import IEngine
from catalyst.core.experiment import IExperiment
from catalyst.core.functional import filter_callbacks_by_node, sort_callbacks_by_order
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.settings import SETTINGS
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

logger = logging.getLogger(__name__)


BATCH_METRICS = Dict[str, float]
LOADER_METRICS = Dict[str, BATCH_METRICS]
EPOCH_METRICS = Dict[str, LOADER_METRICS]


@lru_cache(maxsize=42)
def _has_str_intersections(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


class RunnerException(Exception):
    """Exception class for all runner errors."""

    pass


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
    """

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        # device: Device = None,
    ):
        """
        Args:
            model: Torch model object
            engine: IEngine instance
            # device: Torch device object
        """
        # the core
        self.model: RunnerModel = model
        # self._device = device
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
        self.loader_sample_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0

        # batch info
        self.batch_size: int = 0

        # extra
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

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
                loader_sample_len=self.loader_sample_len,
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
                loader_sample_len=self.loader_sample_len,
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
        # @TODO: trick to save pure epoch-based metrics, like lr/momentum
        self.epoch_metrics["_epoch_"] = {}

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
        self.loader_batch_size: int = self.loader.batch_size
        self.loader_batch_len: int = len(self.loader)
        self.loader_sample_len: int = len(self.loader.dataset)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.stage_epoch_step)

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
        # for callback in self.callbacks.values():
        #     callback.handle_batch(self)

    def _run_batch(self) -> None:
        self._run_event("on_batch_start")
        # @TODO: handle_batch with Callback?
        self.handle_batch(batch=self.batch)
        # self.handle_batch()
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

    def _run_experiment(self, rank=0, world_size=1) -> None:
        # TODO: move this logic somewhere else
        # NOTE: engine should be built elsewhere but not here
        if isinstance(self.engine, DistributedDataParallelEngine):
            self.engine.device = rank
            self.engine._world_size = world_size

            logger.warning(f"rank: {rank}")
            logger.warning(f"world size: {world_size}")
            logger.warning(f"engine: {self.engine}")

        self._run_event("on_experiment_start")
        for self.stage_key in self.experiment.stages:
            # if self.engine.rank < 0:
            #     # single-device branch (cpu, gpu, dp)
            self._run_stage()
            # else:
            #     # ddp-device branch
            #     torch.multiprocessing.spawn(
            #         self._run_stage, args=(), nprocs=self.engine.world_size
            #     )
            #     # raise NotImplementedError()
        self._run_event("on_experiment_end")

    def _run_ddp_experiment(self) -> None:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            self._run_experiment, args=(world_size,), nprocs=world_size, join=True,
        )

    def run(self, experiment: IExperiment = None) -> "IRunner":
        """
        Starts the experiment.

        Args:
            experiment: Experiment instance to use for Runner.

        Returns:
            self, `IRunner` instance after the experiment
        """
        self.experiment = experiment or self.experiment
        try:
            # @TODO: we need to move it to _run_experiment
            # as we can understand Engine only after `on_experiment_start`
            if isinstance(self.engine, DistributedDataParallelEngine):
                self._run_ddp_experiment()
            else:
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

        # seed fix during dataset creation for reproducibility
        set_global_seed(self.experiment.seed + self.engine.rank + self.global_epoch_step)
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
        # seed fix during components creation for reproducibility
        set_global_seed(self.experiment.seed + self.global_epoch_step)
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=model_fn,
            criterion_fn=partial(self.experiment.get_criterion, stage=self.stage_key),
            optimizer_fn=partial(self.experiment.get_optimizer, stage=self.stage_key),
            scheduler_fn=partial(self.experiment.get_scheduler, stage=self.stage_key),
        )

        # @TODO: we could refactor here
        migrate_callbacks_from_previous_stage = stage_params.get(
            "migrate_callbacks_from_previous_stage", True
        )
        # seed fix during callbacks creation for reproducibility
        set_global_seed(self.experiment.seed + self.engine.rank + self.global_epoch_step)
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
                raise RunnerException(f"DataLoader with name {loader_key} is empty.")

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
            raise NotImplementedError(f"DataLoader with name {self.loader_key} is empty.")
        set_global_seed(self.experiment.seed + self.engine.rank + self.global_epoch_step)

    def on_stage_end(self, runner: "IRunner") -> None:
        # clean process if DDP training or do nothing
        self.engine.deinit_components()


__all__ = ["IRunner", "IStageBasedRunner", "RunnerException"]
