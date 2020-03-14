from typing import Any, Dict, Iterable, List, Mapping, Union  # isort:skip
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader

from catalyst.dl import Callback, Experiment, utils
from catalyst.utils.tools.typing import Criterion, Model, Optimizer, Scheduler


class BaseExperiment(Experiment):
    """
    Super-simple one-staged experiment
        you can use to declare experiment in code
    """
    def __init__(
        self,
        model: Model,
        loaders: "OrderedDict[str, DataLoader]",
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
        check_run: bool = False,
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        distributed_params: Dict = None,
        monitoring_params: Dict = None,
        initial_seed: int = 42,
    ):
        """
        Args:
            model (Model): model
            loaders (dict): dictionary containing one or several
                ``torch.utils.data.DataLoader`` for training and validation
            callbacks (List[catalyst.dl.Callback]): list of callbacks
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
            verbose (bool): ff true, it displays the status of the training
                to the console.
            state_kwargs (dict): additional state params to ``State``
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            distributed_params (dict): dictionary with the parameters
                for distributed and FP16 method
            monitoring_params (dict): dict with the parameters
                for monitoring services
            initial_seed (int): experiment's initial seed value
        """
        self._model = model
        self._loaders = loaders
        self._callbacks = utils.process_callbacks(callbacks)

        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._initial_seed = initial_seed
        self._logdir = logdir
        self._stage = stage
        self._num_epochs = num_epochs
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric
        self._verbose = verbose
        self._check_run = check_run
        self._additional_state_kwargs = state_kwargs or {}
        self._checkpoint_data = checkpoint_data or {}
        self._distributed_params = distributed_params or {}
        self._monitoring_params = monitoring_params or {}

    @property
    def initial_seed(self) -> int:
        """Experiment's initial seed value"""
        return self._initial_seed

    @property
    def logdir(self):
        """Path to the directory where the experiment logs"""
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        """Experiment's stage names (array with one value)"""
        return [self._stage]

    @property
    def distributed_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 method"""
        return self._distributed_params

    @property
    def monitoring_params(self) -> Dict:
        """Dict with the parameters for monitoring services"""
        return self._monitoring_params

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage"""
        default_params = dict(
            logdir=self.logdir,
            num_epochs=self._num_epochs,
            valid_loader=self._valid_loader,
            main_metric=self._main_metric,
            verbose=self._verbose,
            minimize_metric=self._minimize_metric,
            checkpoint_data=self._checkpoint_data,
        )
        state_params = {**default_params, **self._additional_state_kwargs}
        return state_params

    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage"""
        return self._model

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage"""
        return self._criterion

    def get_optimizer(self, stage: str, model: nn.Module) -> Optimizer:
        """Returns the optimizer for a given stage"""
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> Scheduler:
        """Returns the scheduler for a given stage"""
        return self._scheduler

    def get_loaders(
        self,
        stage: str,
        epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage"""
        return self._loaders

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage"""
        return self._callbacks


__all__ = ["BaseExperiment"]
