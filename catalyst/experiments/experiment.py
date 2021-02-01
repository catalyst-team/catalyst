from typing import Any, Dict, Iterable, List, Mapping, Tuple, TYPE_CHECKING, Union
from collections import OrderedDict
import warnings

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, VerboseCallback
from catalyst.core.experiment import IExperiment
from catalyst.core.functional import check_callback_isinstance, sort_callbacks_by_order
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.engines import DeviceEngine, IEngine
from catalyst.settings import IS_CUDA_AVAILABLE, SETTINGS
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from catalyst.utils.loaders import get_loaders_from_params

if TYPE_CHECKING:
    from catalyst.core.callback import Callback


def _get_default_hparams(experiment: "Experiment"):
    return {}


def _get_default_engine():
    return DeviceEngine("cuda" if IS_CUDA_AVAILABLE else "cpu")


def _process_loaders(
    loaders: "OrderedDict[str, DataLoader]", stage: str, valid_loader: str, initial_seed: int,
) -> "Tuple[OrderedDict[str, DataLoader], str]":
    """Prepares loaders for a given stage."""
    if not isinstance(loaders[list(loaders.keys())[0]], DataLoader):
        loaders = get_loaders_from_params(initial_seed=initial_seed, **loaders)
    if not stage.startswith(SETTINGS.stage_infer_prefix):  # train stage
        if len(loaders) == 1:
            valid_loader = list(loaders.keys())[0]
            warnings.warn("Attention, there is only one dataloader - " + str(valid_loader))
        assert (
            valid_loader in loaders
        ), "The validation loader must be present in the loaders used during experiment."
    return loaders, valid_loader


class Experiment(IExperiment):
    """One-staged experiment, you can use it to declare experiments in code."""

    def __init__(
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
        callbacks: "Union[OrderedDict[str, Callback], List[Callback]]" = None,
        # the loggers
        loggers: "Union[Dict[str, Callback]]" = None,
        # experiment info
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        # stage info
        stage: str = "train",
        num_epochs: int = 1,
        # extra info (callbacks info)
        logdir: str = None,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        check_time: bool = False,
        check_run: bool = False,
        overfit: bool = False,
    ):
        """
        Args:
            model: model
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
            stage: current stage
            criterion: criterion function
            optimizer: optimizer
            scheduler: scheduler
            trial : hyperparameters optimization trial.
                Used for integrations with Optuna/HyperOpt/Ray.tune.
            num_epochs: number of experiment's epochs
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric: flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose: if True, it displays the status of the training
                to the console.
            check_time: if True, computes the execution time
                of training process and displays it to the console.
            check_run: if True, we run only 3 batches per loader
                and 3 epochs per stage to check pipeline correctness
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            stage_kwargs: additional stage params
            checkpoint_data: additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            engine_params: dictionary with the parameters
                for distributed and FP16 method
            seed: experiment's initial seed
            engine: engine to use, if ``None`` then will be used
                device engine.
        """
        # the data
        self._loaders = loaders
        # the core
        self._model = model
        self._engine: IEngine = engine
        self._trial = trial
        # the components
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        # the callbacks
        self._callbacks = sort_callbacks_by_order(callbacks)
        # the loggers
        self._loggers = loggers
        # experiment info
        self._seed = seed
        self._hparams = hparams
        # stage info
        self._stage = stage
        self._num_epochs = num_epochs
        # extra info (callbacks info)
        self._logdir = logdir
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric
        self._verbose = verbose
        self._check_time = check_time
        self._check_run = check_run
        self._overfit = overfit

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
            return _get_default_hparams(self)

    @property
    def stages(self) -> Iterable[str]:
        """Experiment's stage names (array with one value)."""
        return [self._stage]

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        return {
            "num_epochs": self._num_epochs,
            "migrate_model_from_previous_stage": False,
            "migrate_callbacks_from_previous_stage": False,
        }

    def get_trial(self) -> ITrial:
        return self._trial

    def get_engine(self) -> IEngine:
        return self._engine or _get_default_engine()

    def get_loggers(self) -> Dict[str, ILogger]:
        return self._loggers or {}

    def get_loaders(self, stage: str, epoch: int = None,) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders, self._valid_loader = _process_loaders(
            loaders=self._loaders,
            stage=self._stage,
            valid_loader=self._valid_loader,
            initial_seed=self._seed,
        )
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

    def get_optimizer(self, stage: str, model: nn.Module) -> Optimizer:
        """Returns the optimizer for a given stage."""
        return (
            self._optimizer(model)
            if callable(self._optimizer) and not isinstance(self._optimizer, optim.Optimizer)
            else self._optimizer
        )

    def get_scheduler(self, stage: str, optimizer=None) -> Scheduler:
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
        """
        Returns the callbacks for a given stage.
        """
        callbacks = self._callbacks or OrderedDict()

        is_already_present = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
        )

        if self._verbose and not is_already_present(VerboseCallback):
            callbacks["_verbose"] = VerboseCallback()
        if self._check_time and not is_already_present(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if self._check_run and not is_already_present(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if self._overfit and not is_already_present(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if not stage.startswith(SETTINGS.stage_infer_prefix):
            if self._logdir is not None and not is_already_present(CheckpointCallback):
                callbacks["_checkpoint"] = CheckpointCallback(
                    logdir=self._logdir,
                    loader_key=self._valid_loader,
                    metric_key=self._main_metric,
                    minimize=self._minimize_metric,
                )

        return callbacks


__all__ = ["Experiment"]
