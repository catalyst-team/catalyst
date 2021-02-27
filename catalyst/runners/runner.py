from typing import Any, Dict, Generator, Iterable, List, Mapping, TYPE_CHECKING, Union
from collections import OrderedDict
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, TqdmCallback
from catalyst.core.callback import Callback
from catalyst.core.functional import check_callback_isinstance, sort_callbacks_by_order
from catalyst.core.logger import ILogger
from catalyst.core.runner import IRunner
from catalyst.core.trial import ITrial
from catalyst.engines import DeviceEngine, IEngine
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.settings import IS_CUDA_AVAILABLE
from catalyst.typing import (
    Criterion,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    Scheduler,
)
from catalyst.utils import check_amp_available
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from catalyst.utils.loaders import get_loaders_from_params
from catalyst.utils.misc import maybe_recursive_call, set_global_seed

if TYPE_CHECKING:
    from catalyst.core.callback import Callback


def _get_default_engine():
    return DeviceEngine("cuda" if IS_CUDA_AVAILABLE else "cpu")


def _process_loaders(
    loaders: "OrderedDict[str, DataLoader]", initial_seed: int
) -> "OrderedDict[str, DataLoader]":
    if not isinstance(loaders[list(loaders.keys())[0]], DataLoader):
        loaders = get_loaders_from_params(initial_seed=initial_seed, **loaders)
    return loaders


class Runner(IRunner):
    """Single-stage deep learning Runner with user-friendly API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the core
        self._model: RunnerModel = self.model
        self._engine: IEngine = self.engine
        # self.experiment: IExperiment = None
        self._trial: ITrial = None
        # the data
        self._loaders: Dict[str, DataLoader] = None
        # the components
        self._criterion: RunnerCriterion = None
        self._optimizer: RunnerOptimizer = None
        self._scheduler: RunnerScheduler = None
        # the callbacks
        self._callbacks: Dict[str, Callback] = {}
        # the loggers
        self._loggers: Dict[str, ILogger] = {}
        # extra
        self._seed = 42
        self._hparams: Dict = None
        self._stage: str = "stage"
        self._num_epochs: int = 1
        self._logdir = None
        self._valid_loader = None
        self._valid_metric = None
        self._minimize_valid_metric = None
        self._verbose = False
        self._timeit = False
        self._check = False
        self._overfit = False
        self._load_best_on_end = False

    @property
    def seed(self) -> int:
        """Experiment's initial seed value."""
        return self._seed

    @property
    def name(self) -> str:
        return "experiment" if self._trial is None else f"experiment_{self._trial.number}"

    @property
    def hparams(self) -> Dict:
        """Returns hyper parameters"""
        if self._hparams is not None:
            return self._hparams
        elif self._trial is not None:
            return self._trial.params
        else:
            return {}

    @property
    def stages(self) -> Iterable[str]:
        """Experiment's stage names (array with one value)."""
        return [self._stage]

    def get_stage_len(self, stage: str) -> int:
        return self._num_epochs

    def get_trial(self) -> ITrial:
        return self._trial

    def get_engine(self) -> IEngine:
        return self._engine or _get_default_engine()

    def get_loggers(self) -> Dict[str, ILogger]:
        loggers = self._loggers or {}
        is_logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not is_logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if self._logdir is not None and not is_logger_exists(CSVLogger):
            loggers["_csv"] = CSVLogger(logdir=self._logdir)
        if self._logdir is not None and not is_logger_exists(TensorboardLogger):
            loggers["_tensorboard"] = TensorboardLogger(
                logdir=os.path.join(self._logdir, "tensorboard")
            )
        return loggers

    def get_loaders(self, stage: str,) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = _process_loaders(loaders=self._loaders, initial_seed=self.seed)
        return self._loaders

    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage."""
        model = (
            self._model()
            if callable(self._model) and not isinstance(self._model, nn.Module)
            else self._model
        )
        return model

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage."""
        return (
            self._criterion()
            if callable(self._criterion) and not isinstance(self._criterion, nn.Module)
            else self._criterion
        )

    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        """Returns the optimizer for a given stage."""
        return (
            self._optimizer(model)
            if callable(self._optimizer) and not isinstance(self._optimizer, optim.Optimizer)
            else self._optimizer
        )

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        """Returns the scheduler for a given stage."""
        return (
            self._scheduler(optimizer)
            if callable(self._scheduler)
            and not isinstance(
                self._scheduler,
                (optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler._LRScheduler),
            )
            else self._scheduler
        )

    def get_callbacks(self, stage: str) -> "OrderedDict[str, ICallback]":
        """Returns the callbacks for a given stage."""
        callbacks = sort_callbacks_by_order(self._callbacks)
        is_callback_exists = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if self._verbose and not is_callback_exists(TqdmCallback):
            callbacks["_verbose"] = TqdmCallback()
        if self._timeit and not is_callback_exists(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if self._check and not is_callback_exists(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if self._overfit and not is_callback_exists(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if self._logdir is not None:
            if not is_callback_exists(ICheckpointCallback):
                callbacks["_checkpoint"] = CheckpointCallback(
                    logdir=os.path.join(self._logdir, "checkpoints"),
                    loader_key=self._valid_loader,
                    metric_key=self._valid_metric,
                    minimize=self._minimize_valid_metric,
                )
            # else:
            #     raise NotImplementedError("CheckpointCallback already exist")

        # if self._valid_metric is not None:
        #     have_required_callback = False
        #     for callback in callbacks.values():
        #         if isinstance(callback, CriterionCallback) and callback.metric_key == self._valid_metric:
        #             have_required_callback = True
        #     assert (
        #         have_required_callback
        #     ), f"No CriterionCallback with `metric_key={self._valid_metric}` were found"
        return callbacks

    def train(
        self,
        *,
        # the data
        loaders: "OrderedDict[str, DataLoader]",
        # the core
        model: Model,
        engine: Union["IEngine", str] = None,
        trial: ITrial = None,
        # the components
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        # the callbacks
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        # the loggers
        loggers: "Dict[str, ILogger]" = None,
        # experiment info
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        # stage info
        num_epochs: int = 1,
        # extra info (callbacks info)
        logdir: str = None,
        # resume: str = None,
        valid_loader: str = None,
        valid_metric: str = None,
        minimize_valid_metric: bool = True,
        verbose: bool = False,
        timeit: bool = False,
        check: bool = False,
        overfit: bool = False,
        load_best_on_end: bool = False,
        # engine extra params, @TODO: what to do with them?
        # fp16: Union[Dict, bool] = None,
        # distributed: bool = False,
    ) -> None:
        """
        Starts the train stage of the model.

        Args:
            model: model to train
            criterion: criterion function for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            datasets (OrderedDict[str, Union[Dataset, Dict, Any]]): dictionary
                with one or several  ``torch.utils.data.Dataset``
                for training, validation or inference
                used for Loaders automatic creation
                preferred way for distributed training setup
            loaders (OrderedDict[str, DataLoader]): dictionary
                with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            callbacks (Union[List[Callback], OrderedDict[str, Callback]]):
                list or dictionary with Catalyst callbacks
            logdir: path to output directory
            resume: path to checkpoint for model
            num_epochs: number of training epochs
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            valid_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_valid_metric: flag to indicate whether
                the ``valid_metric`` should be minimized or not.
            verbose: if `True`, it displays the status of the training
                to the console.
            fp16: parameters for fp16/distributed training.
                to use pytorch native amp - ``{"amp": True}``.
                to use apex - ``{"apex": True, "opt_level": "O1", ...}``.
                If fp16=True, params by default will be:
                ``{"amp": True}`` if torch>=1.6.0,
                ``{"apex": True, "opt_level": "O1", ...}`` if torch<1.6.0.
                See https://nvidia.github.io/apex/amp.html#properties for
                more params.
            distributed: if `True` will start training
                in distributed mode.
                Note: Works only with python scripts. No jupyter support.
            check: if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            timeit: if True, computes the execution time
                of training process and displays it to the console.
            load_best_on_end: if True, Runner will load
                best checkpoint state (model, optimizer, etc)
                according to validation metrics. Requires specified ``logdir``.
            seed: experiment's initial seed value
            state_kwargs: deprecated, use `stage_kwargs` instead

        Raises:
            NotImplementedError: if both `resume` and `CheckpointCallback`
                already exist
        """
        # experiment setup
        self._engine = engine
        self._trial = trial
        self._loggers = loggers
        # the data
        self._loaders = loaders
        # the components
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        # the callbacks
        self._callbacks = callbacks
        # extra
        self._stage = "train"
        self._seed = seed
        self._hparams = hparams
        self._num_epochs = num_epochs
        self._logdir = logdir
        self._valid_loader = valid_loader
        self._valid_metric = valid_metric
        self._minimize_valid_metric = minimize_valid_metric
        self._verbose = verbose
        self._timeit = timeit
        self._check = check
        self._overfit = overfit
        self._load_best_on_end = load_best_on_end
        # run
        self.run()

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            **kwargs: additional kwargs to pass to the model

        # noqa: DAR202
        Returns:
            Mapping[str, Any]: model output dictionary

        Raises:
            NotImplementedError: if not implemented yet
        """
        raise NotImplementedError("Please implement `runner.predict_batch` method")

    @torch.no_grad()
    def predict_loader(
        self,
        *,
        loader: DataLoader,
        model: Model = None,
        resume: str = None,
        # fp16: Union[Dict, bool] = None,
        initial_seed: int = 42,
    ) -> Generator:
        """
        Runs model inference on PyTorch DataLoader and returns
        python generator with model predictions from `runner.predict_batch`.
        Cleans up the experiment info to avoid possible collisions.
        Sets `is_train_loader` and `is_valid_loader` to `False` while
        keeping `is_infer_loader` as True. Moves model to evaluation mode.

        Args:
            loader: loader to predict
            model: model to use for prediction
            resume: path to checkpoint to resume
            fp16 (Union[Dict, bool]): fp16 settings (same as in `train`)
            initial_seed: seed to use before prediction

        Yields:
            bathes with model predictions
        """
        # fp16 = _resolve_bool_fp16(fp16)

        if model is not None:
            self.model = model
        assert self.model is not None

        if resume is not None:
            checkpoint = load_checkpoint(resume)
            unpack_checkpoint(checkpoint, model=self.model)

        # @TODO: we need engine here
        self.model = self.engine.sync_device(self.model)
        maybe_recursive_call(self.model, "train", mode=False)

        set_global_seed(initial_seed)
        for batch in loader:
            yield self.predict_batch(batch)


__all__ = ["Runner"]
