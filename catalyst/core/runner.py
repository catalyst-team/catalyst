from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from functools import lru_cache
import logging

import torch
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from catalyst.core.callback import Callback, ICallback
from catalyst.core.engine import IEngine
from catalyst.core.functional import (
    filter_callbacks_by_node,
    sort_callbacks_by_order,
    validate_loaders,
)
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.typing import (
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
from catalyst.utils.misc import maybe_recursive_call, set_global_seed

LOGGER = logging.getLogger(__name__)


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

            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.engine.IEngine`
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
        # self.experiment: IExperiment = None
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
        self.run_key: str = None
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

    # @TODO: remove hotfix
    @property
    def device(self) -> Device:
        """Returns the runner's device instance."""
        return self.engine.device

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        return 42

    @property
    def name(self) -> str:
        """@TODO: docs"""
        # @TODO: return the hash from the IRunner instance
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
    @abstractmethod
    def stages(self) -> Iterable[str]:
        """Experiment's stage names.

        Example::

            >>> runner.stages
            ["pretraining", "finetuning"]
        """
        pass

    def get_stage_len(self, stage: str) -> int:
        """@TODO: docs"""
        return 1

    def get_trial(self) -> Optional[ITrial]:
        """@TODO: docs"""
        return None  # noqa: WPS324

    @abstractmethod
    def get_engine(self) -> IEngine:
        """@TODO: docs"""
        return None  # noqa: WPS324

    def get_loggers(self) -> Dict[str, ILogger]:
        """@TODO: docs"""
        return {}

    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        """Returns the datasets for a given stage and epoch.  # noqa: DAR401

        .. note::
            For Deep Learning cases you have the same dataset
            during whole stage.

            For Reinforcement Learning it's common to change the dataset
            (experiment) every training epoch.

        Args:
            stage: stage name of interest, like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            OrderedDict[str, Dataset]: Ordered dictionary
                with datasets for current stage and epoch.

        .. note::
            We need ordered dictionary to guarantee the correct dataflow
            and order of our training datasets.
            For example, to run train loader before validation one :)

        Example::

            >>> runner.get_datasets(stage="training")
            OrderedDict({
                "train": CsvDataset(in_csv=in_csv_train, ...),
                "valid": CsvDataset(in_csv=in_csv_valid, ...),
            })


        """
        raise NotImplementedError

    # def get_samplers(self, stage: str = None):
    #     raise NotImplementedError
    #
    # def get_transforms(self, stage: str = None):
    #     """Returns the data transforms for a given stage and dataset.
    #
    #     Args:
    #         stage: stage name of interest,
    #             like "pretrain" / "train" / "finetune" / etc
    #         dataset: dataset name of interest,
    #             like "train" / "valid" / "infer"
    #
    #     .. note::
    #         For datasets/loaders naming please follow
    #         :py:mod:`catalyst.core.runner` documentation.
    #
    #     Returns:  # noqa: DAR202
    #         Data transformations to use for specified dataset.
    #
    #     """
    #     raise NotImplementedError

    @abstractmethod  # noqa: WPS463
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage.  # noqa: DAR401

        .. note::
            Wrapper for
            :py:mod:`catalyst.core.experiment.IExperiment.get_datasets`.
            For most of your experiments you need to rewrite `get_datasets`
            method only.

        Args:
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR201, DAR202
            OrderedDict[str, DataLoader]: Ordered dictionary
                with loaders for current stage and epoch.

        """
        pass

    @abstractmethod  # noqa: WPS463
    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage and epoch.

        Example::

            # suppose we have typical MNIST model, like
            # nn.Sequential(nn.Linear(28*28, 128), nn.Linear(128, 10))
            >>> runner.get_model(stage="train")
            Sequential(
             : Linear(in_features=784, out_features=128, bias=True)
             : Linear(in_features=128, out_features=10, bias=True)
            )

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR201, DAR202
            Model: model for a given stage.
        """
        pass

    def get_criterion(self, stage: str) -> Optional[Criterion]:
        """Returns the criterion for a given stage and epoch.

        Example::

            # for typical classification task
            >>> runner.get_criterion(stage="train")
            nn.CrossEntropyLoss()

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR201, DAR202
            Criterion: criterion for a given stage.
        """
        return None  # noqa: WPS324

    def get_optimizer(self, stage: str, model: Model) -> Optional[Optimizer]:
        """Returns the optimizer for a given stage and model.

        Example::

            >>> runner.get_optimizer(model=model, stage="train")
            torch.optim.Adam(model.parameters())

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc
            model: model to optimize with stage optimizer

        Returns:  # noqa: DAR201, DAR202
            Optimizer: optimizer for a given stage and model.
        """
        return None  # noqa: WPS324

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Optional[Scheduler]:
        """Returns the scheduler for a given stage and optimizer.

        Example::
            >>> runner.get_scheduler(stage="training", optimizer=optimizer)
            torch.optim.lr_scheduler.StepLR(optimizer)

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc
            optimizer: optimizer to schedule with stage scheduler

        Returns:  # noqa: DAR201, DAR202
            Scheduler: scheduler for a given stage and optimizer.
        """
        return None  # noqa: WPS324

    def _get_model(self) -> Model:
        self.model = self.get_model(stage=self.stage_key)
        return self.model

    def _get_criterion(self) -> Criterion:
        self.criterion = self.get_criterion(stage=self.stage_key)
        return self.criterion

    def _get_optimizer(self) -> Optimizer:
        assert self.model is not None, "You need to setup model first"
        self.optimizer = self.get_optimizer(stage=self.stage_key, model=self.model)
        return self.optimizer

    def _get_scheduler(self) -> Scheduler:
        # assert self.optimizer is not None, "You need to setup optimizer first"
        self.scheduler = self.get_scheduler(stage=self.stage_key, optimizer=self.optimizer)
        return self.scheduler

    def get_callbacks(self, stage: str) -> "OrderedDict[str, ICallback]":
        """Returns callbacks for a given stage.

        .. note::
            To learn more about Catalyst Callbacks mechanism, please follow
            :py:mod:`catalyst.core.callback.Callback` documentation.

        .. note::
            We need ordered dictionary to guarantee the correct dataflow
            and order of metrics optimization.
            For example, to compute loss before optimization ;)

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            OrderedDict[str, Callback]: Ordered dictionary  # noqa: DAR202
            with callbacks for current stage.

        Args:
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary
                with callbacks for current stage.
        """
        return {}

    def log_metrics(self, *args, **kwargs) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.log_metrics(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.run_key,
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
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.log_image(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.run_key,
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
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.log_hparams(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.run_key,
            )

    def flush_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.flush_log()

    def close_log(self) -> None:
        """@TODO: docs."""
        for logger in self.loggers.values():
            logger.close_log()

    def _setup_loaders(self) -> None:
        set_global_seed(self.seed + self.engine.rank + self.global_epoch_step)
        loaders = self.get_loaders(stage=self.stage_key)
        loaders = validate_loaders(loaders)
        self.loaders = loaders

    def _setup_components(self) -> None:
        set_global_seed(self.seed + self.engine.rank + self.global_epoch_step)
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=self._get_model,
            criterion_fn=self._get_criterion,
            optimizer_fn=self._get_optimizer,
            scheduler_fn=self._get_scheduler,
        )

    def _setup_callbacks(self):
        set_global_seed(self.seed + self.engine.rank + self.global_epoch_step)
        callbacks = self.get_callbacks(self.stage_key)
        callbacks = filter_callbacks_by_node(callbacks)
        callbacks = sort_callbacks_by_order(callbacks)
        self.callbacks = callbacks

    def on_experiment_start(self, runner: "IRunner"):
        """Event handler."""
        self.run_key = self.name
        self.global_epoch_step: int = 0
        self.global_batch_step: int = 0
        self.global_sample_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

        self.trial = self.get_trial()
        self.engine = self.get_engine()
        self.loggers = self.get_loggers()
        self.log_hparams(hparams=self.hparams)

    def on_stage_start(self, runner: "IRunner"):
        """Event handler."""
        assert self.stage_key is not None
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        self.stage_epoch_len = self.get_stage_len(stage=self.stage_key)
        self.stage_epoch_step: int = 0
        self.stage_batch_step: int = 0
        self.stage_sample_step: int = 0
        self._setup_loaders()
        self._setup_components()
        self._setup_callbacks()

    def on_epoch_start(self, runner: "IRunner"):
        """Event handler."""
        self.global_epoch_step += 1
        self.stage_epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)
        # storage for pure epoch-based metrics, like lr/momentum
        self.epoch_metrics["_epoch_"] = {}

        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise RunnerException(f"DataLoader with name {loader_key} is empty.")
        set_global_seed(self.seed + self.engine.rank + self.global_epoch_step)

    def on_loader_start(self, runner: "IRunner"):
        """Event handler."""
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        assert self.is_train_loader or self.is_valid_loader or self.is_infer_loader
        self.loader_batch_size: int = self.loader.batch_size
        self.loader_batch_len: int = len(self.loader)
        self.loader_sample_len: int = len(self.loader.dataset)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        if self.loader_batch_len == 0:
            raise NotImplementedError(f"DataLoader with name {self.loader_key} is empty.")
        set_global_seed(self.seed + self.engine.rank + self.global_epoch_step)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.stage_epoch_step)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
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
        """Event handler."""
        self.log_metrics(metrics=self.batch_metrics, scope="batch")

    def on_loader_end(self, runner: "IRunner"):
        """Event handler."""
        self.log_metrics(metrics=self.loader_metrics, scope="loader")
        self.epoch_metrics[self.loader_key] = self.loader_metrics.copy()

    def on_epoch_end(self, runner: "IRunner"):
        """Event handler."""
        self.log_metrics(metrics=self.epoch_metrics, scope="epoch")
        self.flush_log()

    def on_stage_end(self, runner: "IRunner"):
        """Event handler."""
        self.engine.deinit_components()

    def on_experiment_end(self, runner: "IRunner"):
        """Event handler."""
        self.close_log()

    def on_exception(self, runner: "IRunner"):
        """Event handler."""
        raise self.exception

    def _run_event(self, event: str) -> None:
        if _has_str_intersections(event, ("_start")):
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

        # NOTE: wrapped forward because need to scale forward propagation
        # as it was noted in docs:
        #   https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
        # @TODO: handle_batch with Callback?
        with self.engine.autocast():
            self.handle_batch(batch=self.batch)
        # self.handle_batch()
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        self._run_event("on_loader_start")
        with torch.set_grad_enabled(self.is_train_loader):
            for self.loader_batch_step, self.batch in enumerate(self.loader):
                # @TODO: could we move `with self.engine.autocast():` here?
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

    def _run_stage(self, rank=0, world_size=1) -> None:
        # TODO: move this logic somewhere else
        # NOTE: engine should be built elsewhere but not here
        if isinstance(self.engine, DistributedDataParallelEngine):
            self.engine._rank = rank
            # self.engine._world_size = world_size

            LOGGER.warning(f"rank: {rank}")
            LOGGER.warning(f"world size: {world_size}")
            LOGGER.warning(f"engine: {self.engine}")

        self._run_event("on_stage_start")
        while self.stage_epoch_step < self.stage_epoch_len:
            self._run_epoch()
            if self.need_early_stop:
                self.need_early_stop = False
                break
        self._run_event("on_stage_end")

    def _run_experiment(self) -> None:
        self._run_event("on_experiment_start")
        for self.stage_key in self.stages:
            if self.engine.rank < 0:
                # single-device branch (cpu, gpu, dp)
                self._run_stage()
            else:
                # ddp-device branch
                world_size = self.engine.world_size
                torch.multiprocessing.spawn(
                    self._run_stage, args=(world_size), nprocs=world_size, join=True,
                )
                # raise NotImplementedError()
        self._run_event("on_experiment_end")

    def run(self) -> "IRunner":
        """Runs the experiment.

        Returns:
            self, `IRunner` instance after the experiment
        """
        try:
            self._run_experiment()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


__all__ = ["IRunner", "RunnerException"]
