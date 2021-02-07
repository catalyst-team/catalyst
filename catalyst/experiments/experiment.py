from typing import Any, Dict, Iterable, List, Mapping, TYPE_CHECKING, Union
from collections import OrderedDict

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst.core.experiment import IExperiment
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.engines import DeviceEngine, IEngine
from catalyst.settings import IS_CUDA_AVAILABLE
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from catalyst.utils.loaders import get_loaders_from_params

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
        loggers: "Union[Dict[str, ILogger]]" = None,
        # experiment info
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        # stage info
        stage: str = "train",
        num_epochs: int = 1,
    ):
        """
        Args:
            loaders (OrderedDict[str, DataLoader]): dictionary
                with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            model: model
            engine: engine to use, if ``None`` then will be used device engine.
            trial : hyperparameters optimization trial.
                Used for integrations with Optuna/HyperOpt/Ray.tune.
            criterion: criterion function
            optimizer: optimizer
            scheduler: scheduler
            callbacks: list or dictionary with Catalyst callbacks
            loggers: dictionary with Catalyst loggers
            seed: experiment's initial seed
            hparams: dictionary with hyperparameters
            stage: stage name
            num_epochs: number of experiment's epochs

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
        self._callbacks = callbacks or OrderedDict()
        # the loggers
        self._loggers = loggers
        # experiment info
        self._seed = seed
        self._hparams = hparams
        # stage info
        self._stage = stage
        self._num_epochs = num_epochs

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
        self._loaders = _process_loaders(loaders=self._loaders, initial_seed=self._seed,)
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
        """Returns the callbacks for a given stage."""
        return self._callbacks


__all__ = ["Experiment"]
