from typing import Any, Dict, Mapping, Optional
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core.callback import Callback, ICallback
from catalyst.core.engine import Engine
from catalyst.core.logger import ILogger
from catalyst.core.misc import (
    check_callbacks,
    get_loader_batch_size,
    get_loader_num_samples,
    is_str_intersections,
    sort_callbacks_by_order,
)
from catalyst.typing import (
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
)
from catalyst.utils.misc import maybe_recursive_call, set_global_seed

BATCH_METRICS = Dict[str, float]  # {"loss": 1.7}
LOADER_METRICS = Dict[str, float]  # {"loss": 1.7}
# {"train": {"loss": 1.7}, "valid": {"loss": 1.7}}
EPOCH_METRICS = Dict[str, LOADER_METRICS]
# {0: {"train": {}, "valid": {}}, 1: {...}}
EXPERIMENT_METRICS = Dict[int, EPOCH_METRICS]


class IRunnerError(Exception):
    """Exception class for all runner errors."""

    pass


class IRunner(ICallback, ILogger, ABC):
    """
    An abstraction that contains all the logic of how to run the experiment,
    epochs, loaders and batches.
    Please check examples.

    Args:
        model: Torch model object
        engine: Engine instance

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.runner.Runner`
        - :py:mod:`catalyst.runners.config.ConfigRunner`

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.engine.Engine`
            - :py:mod:`catalyst.core.callback.Callback`

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    """

    def __init__(self, model: RunnerModel = None, engine: Engine = None):
        """Init."""
        self.engine: Engine = engine
        self.loggers: Dict[str, ILogger] = {}
        self.loaders: Dict[str, DataLoader] = None
        self.model: RunnerModel = model
        self.criterion: RunnerCriterion = None
        self.optimizer: RunnerOptimizer = None
        self.scheduler: RunnerScheduler = None
        self.callbacks: Dict[str, Callback] = {}
        # the dataflow - model input/output and other batch tensors
        self.batch: Dict[str, torch.Tensor] = None
        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: BATCH_METRICS = defaultdict(None)
        self.loader_metrics: LOADER_METRICS = defaultdict(None)
        self.epoch_metrics: EPOCH_METRICS = defaultdict(None)
        self.experiment_metrics: EXPERIMENT_METRICS = defaultdict(None)

        # experiment info
        self.epoch_step: int = 0
        self.batch_step: int = 0
        self.sample_step: int = 0
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
        self._local_rank: int = -1
        self._world_size: int = -1

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        return 42

    @property
    def hparams(self) -> OrderedDict:
        """
        Returns hyper-parameters for current run.

        Example::
            >>> runner.hparams
            OrderedDict([('optimizer', 'Adam'),
             ('lr', 0.02),
             ('betas', (0.9, 0.999)),
             ('eps', 1e-08),
             ('weight_decay', 0),
             ('amsgrad', False),
             ('train_batch_size', 32)])

        Returns:
            dictionary with hyperparameters
        """
        return {}

    @property
    def num_epochs(self) -> int:
        """Returns the number of epochs in the experiment."""
        return 1

    def log_artifact(self, *args, **kwargs) -> None:
        """Logs artifact (file like audio, video, csv, etc.) to available loggers."""
        for logger in self.loggers.values():
            logger.log_artifact(*args, **kwargs, runner=self)

    def log_image(self, *args, **kwargs) -> None:
        """Logs image to available loggers."""
        for logger in self.loggers.values():
            logger.log_image(*args, **kwargs, runner=self)

    def log_hparams(self, *args, **kwargs) -> None:
        """Logs hyperparameters to available loggers."""
        for logger in self.loggers.values():
            logger.log_hparams(*args, **kwargs, runner=self)

    def log_metrics(self, *args, **kwargs) -> None:
        """Logs batch, loader and epoch metrics to available loggers."""
        for logger in self.loggers.values():
            logger.log_metrics(*args, **kwargs, runner=self)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        for logger in self.loggers.values():
            logger.flush_log()

    def close_log(self) -> None:
        """Closes the loggers."""
        for logger in self.loggers.values():
            logger.close_log()

    @abstractmethod
    def get_engine(self) -> Engine:
        """Returns the engine for the experiment."""
        pass

    def get_loggers(self) -> Dict[str, ILogger]:
        """Returns the loggers for the experiment."""
        return {}

    @abstractmethod
    def get_loaders(self) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for the experiment."""
        pass

    @abstractmethod
    def get_model(self) -> RunnerModel:
        """Returns the model for the experiment."""
        pass

    def get_criterion(self) -> Optional[RunnerCriterion]:
        """Returns the criterion for the experiment."""
        return None

    def get_optimizer(self, model: RunnerModel) -> Optional[RunnerOptimizer]:
        """Returns the optimizer for the experiment."""
        return None

    def get_scheduler(self, optimizer: RunnerOptimizer) -> Optional[RunnerScheduler]:
        """Returns the scheduler for the experiment."""
        return None

    def get_callbacks(self) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for the experiment."""
        return {}

    def _setup_loaders(self) -> None:
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)
        loaders = self.get_loaders()
        self.loaders = {
            key: self.engine.prepare(value) for key, value in loaders.items()
        }

    def _setup_model(self) -> RunnerModel:
        self.model = self.get_model()
        return self.model

    def _setup_criterion(self) -> RunnerCriterion:
        self.criterion = self.get_criterion()
        return self.criterion

    def _setup_optimizer(self, model: RunnerModel = None) -> RunnerOptimizer:
        if model is not None:
            self.model = model
        self.optimizer = self.get_optimizer(model=self.model)
        return self.optimizer

    def _setup_scheduler(self, optimizer: RunnerOptimizer = None) -> RunnerScheduler:
        if optimizer is not None:
            self.optimizer = optimizer
        self.scheduler = self.get_scheduler(optimizer=self.optimizer)
        return self.scheduler

    def _setup_components(self) -> None:
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer(model=self.model)
        self.scheduler = self._setup_scheduler(optimizer=self.optimizer)

        if isinstance(self.model, torch.nn.Module):
            self.model = self.engine.prepare(self.model)
        elif isinstance(self.model, dict):
            self.model = {k: self.engine.prepare(v) for k, v in self.model.items()}
        else:
            raise NotImplementedError()

        if isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer = self.engine.prepare(self.optimizer)
        elif isinstance(self.optimizer, dict):
            self.optimizer = {
                k: self.engine.prepare(v) for k, v in self.optimizer.items()
            }
        elif self.optimizer is None:
            pass
        else:
            raise NotImplementedError()

    def _setup_callbacks(self):
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)
        self.callbacks = sort_callbacks_by_order(self.get_callbacks())
        check_callbacks(self.callbacks, self.criterion, self.optimizer, self.scheduler)

    def on_experiment_start(self, runner: "IRunner"):
        """Event handler."""
        self.epoch_step: int = 0
        self.batch_step: int = 0
        self.sample_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False

        # self.engine = self.get_engine()
        self.engine.setup(local_rank=self._local_rank, world_size=self._world_size)
        if self.engine.is_main_process:
            self.loggers = self.get_loggers()
            self.log_hparams(hparams=self.hparams)
        with self.engine.local_main_process_first():
            self._setup_loaders()
        self._setup_components()
        self._setup_callbacks()

    def on_epoch_start(self, runner: "IRunner"):
        """Event handler."""
        self.epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)
        # storage for pure epoch-based metrics, like lr/momentum
        self.epoch_metrics["_epoch_"] = {}

        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise IRunnerError(f"DataLoader with name {loader_key} is empty.")
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)

    def on_loader_start(self, runner: "IRunner"):
        """Event handler."""
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        assert self.is_train_loader or self.is_valid_loader or self.is_infer_loader
        self.loader_batch_size: int = get_loader_batch_size(self.loader)
        self.loader_batch_len: int = len(self.loader)
        self.loader_sample_len: int = get_loader_num_samples(self.loader)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        if self.loader_batch_len == 0:
            raise IRunnerError(f"DataLoader with name {self.loader_key} is empty.")
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.epoch_step)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
        if isinstance(self.batch, dict):
            self.batch_size = len(next(iter(self.batch.values())))
        else:
            self.batch_size = len(self.batch[0])

        # we have an batch per each worker...
        self.batch_step += self.engine.num_processes
        self.loader_batch_step += self.engine.num_processes
        self.sample_step += self.batch_size * self.engine.num_processes
        self.loader_sample_step += self.batch_size * self.engine.num_processes
        self.batch_metrics: Dict = defaultdict(None)

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        # batch-metrics sync under ddp setup is too computation heavy
        # if self.engine.distributed_type == DistributedType.NO: # @TODO: recheck
        self.log_metrics(metrics=self.batch_metrics, scope="batch")

    def on_loader_end(self, runner: "IRunner"):
        """Event handler."""
        self.log_metrics(metrics=self.loader_metrics, scope="loader")
        self.epoch_metrics[self.loader_key] = {
            key: float(value) for key, value in self.loader_metrics.items()
        }

    def on_epoch_end(self, runner: "IRunner"):
        """Event handler."""
        self.log_metrics(metrics=self.epoch_metrics, scope="epoch")
        self.experiment_metrics[self.epoch_step] = self.epoch_metrics.copy()
        self.flush_log()

    def on_experiment_end(self, runner: "IRunner"):
        """Event handler."""
        self.flush_log()
        self.close_log()
        self.engine.cleanup()

    def on_exception(self, runner: "IRunner"):
        """Event handler."""
        raise self.exception

    def _run_event(self, event: str) -> None:
        if is_str_intersections(event, ("_start",)):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if is_str_intersections(event, ("_end", "_exception")):
            getattr(self, event)(self)

    @abstractmethod
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer step during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches from DataLoader.
        """
        pass

    def _run_loader(self) -> None:
        with torch.set_grad_enabled(self.is_train_loader):
            for self.batch in self.loader:
                if self.need_early_stop:
                    self.need_early_stop = False
                    break
                self._run_event("on_batch_start")
                self.handle_batch(batch=self.batch)
                self._run_event("on_batch_end")

    def _run_epoch(self) -> None:
        for self.loader_key, self.loader in self.loaders.items():
            self._run_event("on_loader_start")
            self._run_loader()
            self._run_event("on_loader_end")

    def _run_experiment(self) -> None:
        while self.epoch_step < self.num_epochs:
            if self.need_early_stop:
                break
            self._run_event("on_epoch_start")
            self._run_epoch()
            self._run_event("on_epoch_end")

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._local_rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self._run_experiment()
        self._run_event("on_experiment_end")

    def _run(self) -> None:
        self.engine = self.get_engine()
        self.engine.spawn(self._run_local)

    def run(self) -> "IRunner":
        """Runs the experiment.

        Returns:
            self, `IRunner` instance after the experiment
        """
        try:
            self._run()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


__all__ = ["IRunner", "IRunnerError"]
