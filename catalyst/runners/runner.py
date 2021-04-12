from typing import Any, Dict, Generator, Iterable, List, Mapping, Union
from collections import OrderedDict
import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.criterion import CriterionCallback, ICriterionCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, TqdmCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.scheduler import ISchedulerCallback, SchedulerCallback
from catalyst.core.callback import Callback
from catalyst.core.logger import ILogger
from catalyst.core.misc import callback_isinstance, sort_callbacks_by_order
from catalyst.core.runner import IRunner
from catalyst.core.trial import ITrial
from catalyst.data.loader import ILoaderWrapper
from catalyst.engines import IEngine
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.runners.supervised import ISupervisedRunner
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
from catalyst.utils.data import get_loaders_from_params
from catalyst.utils.misc import maybe_recursive_call, set_global_seed
from catalyst.utils.torch import get_available_engine


def _process_loaders(
    loaders: "OrderedDict[str, DataLoader]", initial_seed: int
) -> "OrderedDict[str, DataLoader]":
    if not isinstance(loaders[list(loaders.keys())[0]], (DataLoader, ILoaderWrapper)):
        loaders = get_loaders_from_params(initial_seed=initial_seed, **loaders)
    return loaders


class Runner(IRunner):
    """Single-stage deep learning Runner with user-friendly API.

    Args:
        *args:
        **kwargs:
    """

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        # the core
        self._trial: ITrial = None
        self._engine: IEngine = self.engine
        self._model: RunnerModel = self.model
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
        # model selection
        self._logdir = None
        self._valid_loader = None
        self._valid_metric = None
        self._minimize_valid_metric = None
        # extras
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
        """Returns run name."""
        return "experiment" if self._trial is None else f"experiment_{self._trial.number}"

    @property
    def hparams(self) -> Dict:
        """Returns hyperparameters."""
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
        """Returns the stage length in epochs for a given stage."""
        return self._num_epochs

    def get_trial(self) -> ITrial:
        """Returns the trial for a run."""
        return self._trial

    def get_engine(self) -> IEngine:
        """Returns the engine for a run."""
        return self._engine or get_available_engine()

    def get_loggers(self) -> Dict[str, ILogger]:
        """Returns the logger for a run."""
        loggers = self._loggers or {}
        is_logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not is_logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if self._logdir is not None and not is_logger_exists(CSVLogger):
            loggers["_csv"] = CSVLogger(logdir=self._logdir, use_logdir_postfix=True)
        if self._logdir is not None and not is_logger_exists(TensorboardLogger):
            loggers["_tensorboard"] = TensorboardLogger(
                logdir=self._logdir, use_logdir_postfix=True
            )

        return loggers

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
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

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage."""
        callbacks = sort_callbacks_by_order(self._callbacks)
        is_callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if self._verbose and not is_callback_exists(TqdmCallback):
            callbacks["_verbose"] = TqdmCallback()
        if self._timeit and not is_callback_exists(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if self._check and not is_callback_exists(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if self._overfit and not is_callback_exists(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if self._logdir is not None and not is_callback_exists(ICheckpointCallback):
            callbacks["_checkpoint"] = CheckpointCallback(
                logdir=os.path.join(self._logdir, "checkpoints"),
                loader_key=self._valid_loader,
                metric_key=self._valid_metric,
                minimize=self._minimize_valid_metric,
            )
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
        valid_loader: str = None,
        valid_metric: str = None,
        minimize_valid_metric: bool = True,
        verbose: bool = False,
        timeit: bool = False,
        check: bool = False,
        overfit: bool = False,
        load_best_on_end: bool = False,
        # engine extra params,
        fp16: bool = False,
        amp: bool = False,
        apex: bool = False,
        ddp: bool = False,
    ) -> None:
        """
        Starts the train stage of the model.

        Args:
            loaders: dictionary with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            model: model to train
            engine: engine to use for model training
            trial: trial to use during model training
            criterion: criterion function for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            callbacks: list or dictionary with Catalyst callbacks
            loggers: dictionary with Catalyst loggers
            seed: experiment's initial seed value
            hparams: hyperparameters for the run
            num_epochs: number of training epochs
            logdir: path to output directory
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            valid_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_valid_metric: flag to indicate whether
                the ``valid_metric`` should be minimized or not (default: True).
            verbose: if `True`, it displays the status of the training to the console.
            timeit: if True, computes the execution time
                of training process and displays it to the console.
            check: if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            load_best_on_end: if True, Runner will load
                best checkpoint state (model, optimizer, etc)
                according to validation metrics. Requires specified ``logdir``.
            fp16: boolean flag to use half-precision training (AMP > APEX)
            amp: boolean flag to use amp half-precision
            apex: boolean flag to use apex half-precision
            ddp: if `True` will start training in distributed mode.
                Note: Works only with python scripts. No jupyter support.
        """
        # experiment setup
        self._engine = engine or get_available_engine(fp16=fp16, ddp=ddp, amp=amp, apex=apex)
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
            batch: dictionary with data batches from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping: model output dictionary

        Raises:
            NotImplementedError: if not implemented yet
        """
        raise NotImplementedError("Please implement `runner.predict_batch` method")
        return None  # noqa: WPS427

    @torch.no_grad()
    def predict_loader(
        self,
        *,
        loader: DataLoader,
        model: Model = None,
        engine: Union["IEngine", str] = None,
        seed: int = 42,
        # engine extra params,
        fp16: bool = False,
        amp: bool = False,
        apex: bool = False,
        ddp: bool = False,
    ) -> Generator:
        """
        Runs model inference on PyTorch DataLoader and returns
        python generator with model predictions from `runner.predict_batch`.

        Args:
            loader: loader to predict
            model: model to use for prediction
            engine: engine to use for prediction
            seed: random seed to use before prediction
            fp16: boolean flag to use half-precision training (AMP > APEX)
            amp: boolean flag to use amp half-precision
            apex: boolean flag to use apex half-precision
            ddp: if `True` will start training in distributed mode.
                Note: Works only with python scripts. No jupyter support.

        Yields:
            bathes with model predictions
        """
        self._engine = engine or get_available_engine(fp16=fp16, ddp=ddp, amp=amp, apex=apex)

        if model is not None:
            self.model = model
        assert self.model is not None

        # if resume is not None:
        #     checkpoint = load_checkpoint(resume)
        #     unpack_checkpoint(checkpoint, model=self.model)

        self.model = self.engine.sync_device(self.model)
        maybe_recursive_call(self.model, "train", mode=False)

        set_global_seed(seed)
        for batch in loader:
            yield self.predict_batch(batch)


class SupervisedRunner(ISupervisedRunner, Runner):
    """Runner for experiments with supervised model.

    Args:
        model: Torch model instance
        engine: IEngine instance
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output
    """

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """Init."""
        ISupervisedRunner.__init__(
            self,
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            loss_key=loss_key,
        )
        Runner.__init__(self, model=model, engine=engine)

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        .. warning::
            You should not override this method. If you need specific model
            call, override forward() method

        Args:
            batch: dictionary with data batch from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        batch = self._process_batch(batch)
        batch = self.engine.sync_device(tensor_or_module=batch)
        output = self.forward(batch, **kwargs)
        return output

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Prepares the callbacks for selected stage.

        Args:
            stage: stage name

        Returns:
            dictionary with stage callbacks
        """
        callbacks = super().get_callbacks(stage=stage)
        is_callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if isinstance(self._criterion, Criterion) and not is_callback_exists(ICriterionCallback):
            callbacks["_criterion"] = CriterionCallback(
                input_key=self._output_key, target_key=self._target_key, metric_key=self._loss_key,
            )
        if isinstance(self._optimizer, Optimizer) and not is_callback_exists(IOptimizerCallback):
            callbacks["_optimizer"] = OptimizerCallback(metric_key=self._loss_key)
        if isinstance(self._scheduler, (Scheduler, ReduceLROnPlateau)) and not is_callback_exists(
            ISchedulerCallback
        ):
            callbacks["_scheduler"] = SchedulerCallback(
                loader_key=self._valid_loader, metric_key=self._valid_metric
            )
        return callbacks


__all__ = ["Runner", "SupervisedRunner"]
