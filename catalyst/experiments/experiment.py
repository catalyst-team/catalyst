from typing import Any, Dict, Iterable, List, Mapping, Tuple, TYPE_CHECKING, Union
from collections import OrderedDict
import warnings

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.early_stop import CheckRunCallback
from catalyst.callbacks.exception import ExceptionCallback
from catalyst.callbacks.logging import ConsoleLogger, TensorboardLogger, VerboseLogger
from catalyst.callbacks.metric import MetricManagerCallback
from catalyst.callbacks.timer import TimerCallback
from catalyst.callbacks.validation import ValidationManagerCallback
from catalyst.core.experiment import IExperiment
from catalyst.core.functional import check_callback_isinstance, sort_callbacks_by_order
from catalyst.engines import IEngine, process_engine
from catalyst.settings import SETTINGS
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from catalyst.utils.loaders import get_loaders_from_params

if TYPE_CHECKING:
    from catalyst.core.callback import Callback


class Experiment(IExperiment):
    """One-staged experiment, you can use it to declare experiments in code."""

    def __init__(
        self,
        model: Model,
        datasets: "OrderedDict[str, Union[Dataset, Dict, Any]]" = None,
        loaders: "OrderedDict[str, DataLoader]" = None,
        callbacks: "Union[OrderedDict[str, Callback], List[Callback]]" = None,
        logdir: str = None,
        stage: str = "train",
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        trial: Any = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        check_time: bool = False,
        check_run: bool = False,
        overfit: bool = False,
        stage_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        engine_params: Dict = None,
        initial_seed: int = 42,
        engine: str = None,
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
            initial_seed: experiment's initial seed value
            engine: engine to use, if ``None`` then will be used
                device engine.
        """
        assert datasets is not None or loaders is not None, "Please specify the data sources"

        self._engine: IEngine = process_engine(engine)

        self._model = model
        self._loaders, self._valid_loader = self._get_loaders(
            loaders=loaders,
            datasets=datasets,
            stage=stage,
            valid_loader=valid_loader,
            initial_seed=initial_seed,
        )
        self._callbacks = sort_callbacks_by_order(callbacks)

        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._trial = trial

        self._initial_seed = initial_seed
        self._logdir = logdir
        self._stage = stage
        self._num_epochs = num_epochs
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric
        self._verbose = verbose
        self._check_time = check_time
        self._check_run = check_run
        self._overfit = overfit
        self._stage_kwargs = stage_kwargs or {}
        self._checkpoint_data = checkpoint_data or {}
        self._engine_params = engine_params or {}

    @property
    def seed(self) -> int:
        """Experiment's initial seed value."""
        return self._initial_seed

    @property
    def name(self) -> str:
        return "Experiment"

    # @property
    # def logdir(self):
    #     """Path to the directory where the experiment logs."""
    #     return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        """Experiment's stage names (array with one value)."""
        return [self._stage]

    @property
    def hparams(self) -> OrderedDict:
        """Returns hyper parameters"""
        hparams = OrderedDict()
        if self._optimizer is not None:
            optimizer = self._optimizer
            hparams["optimizer"] = optimizer.__repr__().split()[0]
            params_dict = optimizer.state_dict()["param_groups"][0]
            for k, v in params_dict.items():
                if k != "params":
                    hparams[k] = v
        loaders = self.get_loaders(self._stage)
        for k, v in loaders.items():
            if k.startswith("train"):
                hparams[f"{k}_batch_size"] = v.batch_size
        return hparams

    @property
    def trial(self) -> Any:
        """
        Returns hyperparameter trial for current experiment.
        Could be usefull for Optuna/HyperOpt/Ray.tune
        hyperparameters optimizers.

        Returns:
            trial

        Example::

            >>> experiment.trial
            optuna.trial._trial.Trial  # Optuna variant
        """
        return self._trial

    @property
    def engine_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 method."""
        return self._engine_params

    @staticmethod
    def _get_loaders(
        loaders: "OrderedDict[str, DataLoader]",
        datasets: Dict,
        stage: str,
        valid_loader: str,
        initial_seed: int,
    ) -> "Tuple[OrderedDict[str, DataLoader], str]":
        """Prepares loaders for a given stage."""
        if datasets is not None:
            loaders = get_loaders_from_params(initial_seed=initial_seed, **datasets,)
        if not stage.startswith(SETTINGS.stage_infer_prefix):  # train stage
            if len(loaders) == 1:
                valid_loader = list(loaders.keys())[0]
                warnings.warn("Attention, there is only one dataloader - " + str(valid_loader))
            assert valid_loader in loaders, (
                "The validation loader must be present " "in the loaders used during experiment."
            )
        return loaders, valid_loader

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        default_params = {
            "logdir": self.logdir,
            "num_epochs": self._num_epochs,
            "valid_loader": self._valid_loader,
            "main_metric": self._main_metric,
            "verbose": self._verbose,
            "minimize_metric": self._minimize_metric,
            "checkpoint_data": self._checkpoint_data,
        }
        stage_params = {**default_params, **self._stage_kwargs}
        return stage_params

    @property
    def engine(self):
        return self._engine

    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage."""
        # TODO: force user to return model from this method
        model = (
            self._model()
            if callable(self._model) and not isinstance(self._model, nn.Module)
            else self._model
        )
        return self._engine.to_device(model)

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage."""
        # TODO: force user to return criterion from this method
        return self._criterion

    def get_optimizer(self, stage: str, model: nn.Module) -> Optimizer:
        """Returns the optimizer for a given stage."""
        # TODO: force user to return optimizer from this method
        return (
            self._optimizer(model.parameters())
            if callable(self._optimizer) and not isinstance(self._optimizer, optim.Optimizer)
            else self._optimizer
        )

    def get_scheduler(self, stage: str, optimizer=None) -> Scheduler:
        """Returns the scheduler for a given stage."""
        # TODO: force user to return scheduler from this method
        return self._scheduler(optimizer) if callable(self._scheduler) else self._scheduler

    def get_loaders(self, stage: str, epoch: int = None,) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        return self._loaders

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """
        Returns the callbacks for a given stage.
        """
        callbacks = self._callbacks or OrderedDict()
        default_callbacks = []

        if self._verbose:
            default_callbacks.append(("_verbose", VerboseLogger))
        if self._check_time:
            default_callbacks.append(("_timer", TimerCallback))
        if self._check_run:
            default_callbacks.append(("_check", CheckRunCallback))
        if self._overfit:
            default_callbacks.append(("_overfit", BatchOverfitCallback))

        if not stage.startswith("infer"):
            default_callbacks.append(("_metrics", MetricManagerCallback))
            default_callbacks.append(("_validation", ValidationManagerCallback))
            default_callbacks.append(("_console", ConsoleLogger))
            if self.logdir is not None:
                default_callbacks.append(("_saver", CheckpointCallback))
                default_callbacks.append(("_tensorboard", TensorboardLogger))
        default_callbacks.append(("_exception", ExceptionCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                check_callback_isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["Experiment"]
