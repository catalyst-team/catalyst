from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from functools import lru_cache
import logging
import warnings

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

from catalyst.core._misc import (
    callback_isinstance,
    filter_callbacks_by_node,
    sort_callbacks_by_order,
    validate_loaders,
)
from catalyst.core.callback import (
    Callback,
    ICallback,
    ICriterionCallback,
    IOptimizerCallback,
    ISchedulerCallback,
)
from catalyst.core.engine import IEngine
from catalyst.core.logger import ILogger
from catalyst.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    Sampler,
    Scheduler,
)
from catalyst.utils.misc import maybe_recursive_call, set_global_seed

LOGGER = logging.getLogger(__name__)


BATCH_METRICS = Dict[str, float]  # {"loss": 1.7}
LOADER_METRICS = Dict[str, float]  # {"loss": 1.7}
EPOCH_METRICS = Dict[str, LOADER_METRICS]  # {"train": {"loss": 1.7}, "valid": {"loss": 1.7}}
EXPERIMENT_METRICS = Dict[int, EPOCH_METRICS]  # {0: {"train": {}, "valid": {}}, 1: {...}}


@lru_cache(maxsize=42)
def _has_str_intersections(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


def _get_batch_size(loader: DataLoader):
    batch_size = loader.batch_size
    if batch_size is not None:
        return batch_size

    batch_size = loader.batch_sampler.batch_size
    if batch_size is not None:
        return batch_size
    raise NotImplementedError(
        "No `batch_size` found,"
        "please specify it with `loader.batch_size`, or `loader.batch_sampler.batch_size`"
    )


def _get_num_samples(loader: DataLoader):
    batch_size = _get_batch_size(loader)
    if isinstance(loader.batch_sampler, BatchSampler):
        # pytorch default item-based samplers
        if loader.drop_last:
            return (len(loader.dataset) // batch_size) * batch_size
        else:
            return len(loader.dataset)
    else:
        # pytorch batch-based samplers
        return len(loader) * batch_size


class IRunnerError(Exception):
    """Exception class for all runner errors."""

    pass


class IRunner(ICallback, ILogger, ABC):
    """
    An abstraction that contains all the logic of how to run the experiment,
    stages, epochs, loaders and batches.

    IRunner supports the logic for deep learning pipeline configuration with pure python code.
    Please check the examples for intuition.

    Args:
        model: Torch model object
        engine: IEngine instance

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.runner.Runner`
        - :py:mod:`catalyst.runners.config.ConfigRunner`
        - :py:mod:`catalyst.runners.hydra.HydraRunner`

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.engine.IEngine`
            - :py:mod:`catalyst.core.callback.Callback`

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples

    """

    def __init__(self, model: RunnerModel = None, engine: IEngine = None):
        """Init."""
        # the core
        self.model: RunnerModel = model
        self.engine: IEngine = engine
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
        self.batch: Dict[str, torch.Tensor] = None

        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: BATCH_METRICS = defaultdict(None)
        self.loader_metrics: LOADER_METRICS = defaultdict(None)
        self.epoch_metrics: EPOCH_METRICS = defaultdict(None)

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
        self._rank: int = -1
        self._world_size: int = -1

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        return 42

    @property
    def run_key(self) -> str:
        """Returns run name for monitoring tools."""
        return "IRunner"

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
        """Returns number of epochs for an experiment."""
        return 1

    @property
    def _log_defaults(self) -> Dict:
        # TODO: add rank and other dist params here
        return {
            # experiment info
            "run_key": self.run_key,
            "epoch_step": self.epoch_step,
            "batch_step": self.batch_step,
            "sample_step": self.sample_step,
            # loader info
            "loader_key": self.loader_key,
            "loader_batch_len": self.loader_batch_len,
            "loader_sample_len": self.loader_sample_len,
            "loader_batch_step": self.loader_batch_step,
            "loader_sample_step": self.loader_sample_step,
        }

    @abstractmethod
    def get_engine(self) -> IEngine:
        """Returns the engine for the run."""
        return None

    def get_loggers(self) -> Dict[str, ILogger]:
        """Returns the loggers for the run."""
        return {}

    @abstractmethod
    def get_loaders(self) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for an experiment.

        Returns:  # noqa: DAR201, DAR202
            OrderedDict[str, DataLoader]: Ordered dictionary
                with loaders for current stage and epoch.

        """
        pass

    @abstractmethod
    def get_model(self) -> Model:
        """Returns the model for an experiment.

        Example::

            # suppose we have typical MNIST model, like
            # nn.Sequential(nn.Linear(28*28, 128), nn.Linear(128, 10))
            >>> runner.get_model()
            Sequential(
             : Linear(in_features=784, out_features=128, bias=True)
             : Linear(in_features=128, out_features=10, bias=True)
            )

        Returns:  # noqa: DAR201, DAR202
            Model: model for a given stage.
        """
        pass

    def get_criterion(self) -> Optional[Criterion]:
        """Returns the criterion for an experiment.

        Example::

            # for typical classification task
            >>> runner.get_criterion()
            nn.CrossEntropyLoss()

        Returns:  # noqa: DAR201, DAR202
            Criterion: criterion for a given stage.
        """
        return None

    def get_optimizer(self, model: Model) -> Optional[Optimizer]:
        """Returns the optimizer for a model.

        Example::

            >>> runner.get_optimizer(model=model)
            torch.optim.Adam(model.parameters())

        Args:
            model: model to optimize with stage optimizer

        Returns:  # noqa: DAR201, DAR202
            Optimizer: optimizer for a given stage and model.
        """
        return None

    def get_scheduler(self, optimizer: Optimizer) -> Optional[Scheduler]:
        """Returns the scheduler for an optimizer.

        Example::
            >>> runner.get_scheduler(optimizer=optimizer)
            torch.optim.lr_scheduler.StepLR(optimizer)

        Args:
            optimizer: optimizer to schedule with stage scheduler

        Returns:  # noqa: DAR201, DAR202
            Scheduler: scheduler for a given stage and optimizer.
        """
        return None

    def _get_model(self) -> Model:
        self.model = self.get_model()
        return self.model

    def _get_criterion(self) -> Criterion:
        self.criterion = self.get_criterion()
        return self.criterion

    def _get_optimizer(self, model: Model = None) -> Optimizer:
        if model is not None:
            self.model = model
        # assert self.model is not None, "You need to setup model first"
        self.optimizer = self.get_optimizer(model=self.model)
        return self.optimizer

    def _get_scheduler(self, optimizer: Optimizer = None) -> Scheduler:
        if optimizer is not None:
            self.optimizer = optimizer
        # assert self.optimizer is not None, "You need to setup optimizer first"
        self.scheduler = self.get_scheduler(optimizer=self.optimizer)
        return self.scheduler

    def get_callbacks(self) -> "OrderedDict[str, ICallback]":
        """Returns callbacks for an experiment.

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary  # noqa: DAR202
            with callbacks for current stage.
        """
        return {}

    def log_hparams(self, *args, **kwargs) -> None:
        """Logs hyperparameters to available loggers."""
        for logger in self.loggers.values():
            logger.log_hparams(
                *args,
                **kwargs,
                # experiment info
                run_key=self.run_key,
            )

    def log_metrics(self, *args, **kwargs) -> None:
        """Logs batch, loader and epoch metrics to available loggers."""
        for logger in self.loggers.values():
            logger.log_metrics(*args, **kwargs, **self._log_defaults)

    def log_image(self, *args, **kwargs) -> None:
        """Logs image to available loggers."""
        for logger in self.loggers.values():
            logger.log_image(*args, **kwargs, **self._log_defaults)

    def log_artifact(self, *args, **kwargs) -> None:
        """Logs artifact (file like audio, video, csv, etc.) to available loggers."""
        for logger in self.loggers.values():
            logger.log_artifact(*args, **kwargs, **self._log_defaults)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        for logger in self.loggers.values():
            logger.flush_log()

    def close_log(self, *args, **kwargs) -> None:
        """Closes the loggers."""
        for logger in self.loggers.values():
            logger.close_log(*args, **kwargs)

    def _setup_loaders(self) -> None:
        set_global_seed(self.seed + max(0, self.engine.rank) + self.epoch_step)
        loaders = self.get_loaders()
        loaders = {key: self.engine.prepare(value) for key, value in loaders.items()}
        self.loaders = loaders

    def _setup_components(self) -> None:
        set_global_seed(self.seed + max(0, self.engine.rank) + self.epoch_step)
        self.model = self._get_model()
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer(model=self.model)
        self.scheduler = self._get_scheduler(optimizer=self.optimizer)
        self.model, self.optimizer = self.engine.prepare(self.model, self.optimizer)

    def _check_callbacks(self):
        is_callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in self.callbacks.values()
        )
        if isinstance(self.criterion, Criterion) and not is_callback_exists(ICriterionCallback):
            warnings.warn(
                "No ``ICriterionCallback/CriterionCallback`` were found "
                "while runner.criterion is not None."
                "Do you compute the loss during ``runner.handle_batch``?"
            )
        if isinstance(self.optimizer, Optimizer) and not is_callback_exists(IOptimizerCallback):
            warnings.warn(
                "No ``IOptimizerCallback/OptimizerCallback`` were found "
                "while runner.optimizer is not None."
                "Do run backward pass during ``runner.handle_batch``?"
            )
        if isinstance(self.scheduler, (Scheduler, ReduceLROnPlateau)) and not is_callback_exists(
            ISchedulerCallback
        ):
            warnings.warn(
                "No ``ISchedulerCallback/SchedulerCallback`` were found "
                "while runner.scheduler is not None."
                "Do you make scheduler step during ``runner.handle_batch``?"
            )

    def _setup_callbacks(self):
        set_global_seed(self.seed + max(0, self.engine.rank) + self.epoch_step)
        callbacks = self.get_callbacks()
        callbacks = sort_callbacks_by_order(callbacks)
        self.callbacks = callbacks
        self._check_callbacks()

    def on_experiment_start(self, runner: "IRunner"):
        """Event handler."""
        self.epoch_step: int = 0
        self.batch_step: int = 0
        self.sample_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False

        self.engine = self.get_engine()
        self.loggers = self.get_loggers()
        self.log_hparams(hparams=self.hparams)

        if self.engine.is_ddp:
            self.engine.setup_process(rank=self._stage_rank, world_size=self._stage_world_size)
            if not self.engine.is_master_process:
                del self.loggers
                self.loggers = {}

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
        set_global_seed(self.seed + max(0, self.engine.rank) + self.epoch_step)

    def on_loader_start(self, runner: "IRunner"):
        """Event handler."""
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        assert self.is_train_loader or self.is_valid_loader or self.is_infer_loader
        self.loader_batch_size: int = _get_batch_size(self.loader)
        self.loader_batch_len: int = len(self.loader)
        self.loader_sample_len: int = _get_num_samples(self.loader)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        if self.loader_batch_len == 0:
            raise NotImplementedError(f"DataLoader with name {self.loader_key} is empty.")
        set_global_seed(self.seed + max(0, self.engine.rank) + self.epoch_step)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        # @TODO: check this part
        # if isinstance(self.loader.sampler, DistributedSampler):
        #     self.loader.sampler.set_epoch(self.stage_epoch_step)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
        self.batch = self.engine.sync_device(tensor_or_module=self.batch)

        if isinstance(self.batch, dict):
            self.batch_size = len(next(iter(self.batch.values())))
        else:
            self.batch_size = len(self.batch[0])

        # we have an batch per each worker...
        self.global_batch_step += self.engine.world_size
        self.stage_batch_step += self.engine.world_size
        self.loader_batch_step += self.engine.world_size
        self.global_sample_step += self.batch_size * self.engine.world_size
        self.stage_sample_step += self.batch_size * self.engine.world_size
        self.loader_sample_step += self.batch_size * self.engine.world_size
        self.batch_metrics: Dict = defaultdict(None)

    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        # batch-metrics sync under ddp setup is too computation heavy
        if not self.engine.is_ddp:
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
            batch (Mapping[str, Any]): dictionary with data batches from DataLoader.
        """
        pass

    def _run_batch(self) -> None:
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        # NOTE: wrapped forward because need to scale forward propagation
        # as it was noted in docs:
        # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
        self._run_event("on_loader_start")
        with torch.set_grad_enabled(self.is_train_loader):
            for self.batch in self.loader:
                if self.need_early_stop:
                    self.need_early_stop = False
                    break
                self._run_batch()
        self._run_event("on_loader_end")

    def _run_epoch(self) -> None:
        self._run_event("on_epoch_start")
        for self.loader_key, self.loader in self.loaders.items():
            self._run_loader()
        self._run_event("on_epoch_end")

    def _run_experiment(self) -> None:
        while self.epoch_step < self.num_epochs:
            if self.need_early_stop:
                break
            self._run_event("on_epoch_start")
            self._run_epoch()
            self._run_event("on_epoch_end")

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self._run_experiment()
        self._run_event("on_experiment_end")

    def _run(self) -> None:
        # _run_local
        pass

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
