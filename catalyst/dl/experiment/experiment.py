# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union
from collections import OrderedDict
import warnings

from torch import nn
from torch.utils.data import DataLoader, Dataset

from catalyst.core import IExperiment
from catalyst.dl import (
    BatchOverfitCallback,
    Callback,
    CheckpointCallback,
    CheckRunCallback,
    ConsoleLogger,
    ExceptionCallback,
    MetricManagerCallback,
    TensorboardLogger,
    TimerCallback,
    utils,
    ValidationManagerCallback,
    VerboseLogger,
)
from catalyst.dl.utils import check_callback_isinstance
from catalyst.tools import settings
from catalyst.tools.typing import Criterion, Model, Optimizer, Scheduler


class Experiment(IExperiment):
    """
    Super-simple one-staged experiment,
    you can use to declare experiment in code.
    """

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
        distributed_params: Dict = None,
        initial_seed: int = 42,
    ):
        """
        Args:
            model (Model): model
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
            logdir (str): path to output directory
            stage (str): current stage
            criterion (Criterion): criterion function
            optimizer (Optimizer): optimizer
            scheduler (Scheduler): scheduler
            num_epochs (int): number of experiment's epochs
            valid_loader (str): loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric (str): the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric (bool): flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose (bool): if True, it displays the status of the training
                to the console.
            check_time (bool): if True, computes the execution time
                of training process and displays it to the console.
            check_run (bool): if True, we run only 3 batches per loader
                and 3 epochs per stage to check pipeline correctness
            overfit (bool): if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            stage_kwargs (dict): additional stage params
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            distributed_params (dict): dictionary with the parameters
                for distributed and FP16 method
            initial_seed (int): experiment's initial seed value
        """
        assert (
            datasets is not None or loaders is not None
        ), "Please specify the data sources"

        self._model = model
        self._loaders, self._valid_loader = self._get_loaders(
            loaders=loaders,
            datasets=datasets,
            stage=stage,
            valid_loader=valid_loader,
            initial_seed=initial_seed,
        )
        self._callbacks = utils.sort_callbacks_by_order(callbacks)

        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

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
        self._distributed_params = distributed_params or {}

    @property
    def initial_seed(self) -> int:
        """Experiment's initial seed value."""
        return self._initial_seed

    @property
    def logdir(self):
        """Path to the directory where the experiment logs."""
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        """Experiment's stage names (array with one value)."""
        return [self._stage]

    @property
    def distributed_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 method."""
        return self._distributed_params

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
            loaders = utils.get_loaders_from_params(
                initial_seed=initial_seed, **datasets,
            )
        if not stage.startswith(settings.stage_infer_prefix):  # train stage
            if len(loaders) == 1:
                valid_loader = list(loaders.keys())[0]
                warnings.warn(
                    "Attention, there is only one dataloader - "
                    + str(valid_loader)
                )
            assert valid_loader in loaders, (
                "The validation loader must be present "
                "in the loaders used during experiment."
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

    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage."""
        return self._model

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage."""
        return self._criterion

    def get_optimizer(self, stage: str, model: nn.Module) -> Optimizer:
        """Returns the optimizer for a given stage."""
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> Scheduler:
        """Returns the scheduler for a given stage."""
        return self._scheduler

    def get_loaders(
        self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
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
            default_callbacks.append(
                ("_validation", ValidationManagerCallback)
            )
            default_callbacks.append(("_console", ConsoleLogger))
            if self.logdir is not None:
                default_callbacks.append(("_saver", CheckpointCallback))
                default_callbacks.append(("_tensorboard", TensorboardLogger))
        default_callbacks.append(("_exception", ExceptionCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                check_callback_isinstance(x, callback_fn)
                for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["Experiment"]
