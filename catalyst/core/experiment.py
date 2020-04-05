from typing import Any, Dict, Iterable, Mapping, Tuple, Union
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader, Dataset

from catalyst.utils.tools.typing import Criterion, Model, Optimizer, Scheduler

from .callback import Callback


class _Experiment(ABC):
    """
    An abstraction that contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.
    In general, the Experiment knows **what** you would like to run.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment._Experiment`
            - :py:mod:`catalyst.core.runner._Runner`
            - :py:mod:`catalyst.core.state.State`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.dl.experiment.base.BaseExperiment`
        - :py:mod:`catalyst.dl.experiment.config.ConfigExperiment`
        - :py:mod:`catalyst.dl.experiment.gan.GanExperiment`
        - :py:mod:`catalyst.dl.experiment.supervised.SupervisedExperiment`
    """

    @property
    @abstractmethod
    def initial_seed(self) -> int:
        """
        Experiment's initial seed, used to setup `global seed`
        at the beginning of each stage.
        Additionally, Catalyst Runner setups
        `experiment.initial_seed + state.global_epoch + 1`
        as `global seed` each epoch.
        Used for experiment reproducibility.

        Example::

            >>> experiment.initial_seed
            42
        """
        pass

    @property
    @abstractmethod
    def logdir(self) -> str:
        """Path to the directory where the experiment logs would be saved.

        Example::

            >>> experiment.logdir
            ./path/to/my/experiment/logs
        """
        pass

    @property
    @abstractmethod
    def stages(self) -> Iterable[str]:
        """Experiment's stage names.

        Example::

            >>> experiment.stages
            ["pretraining", "training", "finetuning"]

        .. note::
            To understand stages concept, please follow Catalyst documentation,
            for example, :py:mod:`catalyst.core.callback.Callback`
        """
        pass

    @property
    @abstractmethod
    def distributed_params(self) -> Dict:
        """
        Dictionary with the parameters for distributed
        and half-precision training.

        Used in :py:mod:`catalyst.utils.distributed.process_components`
        to setup `Nvidia Apex`_ or `PyTorch distributed`_.

        .. _`Nvidia Apex`: https://github.com/NVIDIA/apex
        .. _`PyTorch distributed`:
            https://pytorch.org/docs/stable/distributed.html

        Example::

            >>> experiment.distributed_params
            {"opt_level": "O1", "syncbn": True}  # Apex variant
        """
        pass

    @property
    @abstractmethod
    def monitoring_params(self) -> Dict:
        """
        Dictionary with the parameters for monitoring services,
        like Alchemy_

        .. _Alchemy: https://alchemy.host

        Example::

            >>> experiment.monitoring_params
            {
                "token": None, # insert your personal token here
                "project": "classification_example",
                "group": "first_trial",
                "experiment": "first_experiment",
            }

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use
            :py:mod:`catalyst.contrib.dl.callbacks.alchemy.AlchemyLogger`
            instead.
        """
        pass

    @abstractmethod
    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        """Returns State parameters for a given stage.

        To learn more about State, please follow
        :py:mod:`catalyst.core.state.State`
        documentation.

        Example::

            >>> experiment.get_state_params(stage="training")
            {
                "logdir": "./logs/training",
                "num_epochs": 42,
                "valid_loader": "valid",
                "main_metric": "loss",
                "minimize_metric": True,
                "checkpoint_data": {"comment": "we are going to make it!"}
            }

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc

        Returns:
            dict: State parameters for a given stage.
        """
        pass

    @abstractmethod
    def get_model(self, stage: str) -> Model:
        """Returns the model for a given stage.

        Example::

            # suppose we have typical MNIST model, like
            # nn.Sequential(nn.Linear(28*28, 128), nn.Linear(128, 10))
            >>> experiment.get_model(stage="training")
            Sequential(
              (0): Linear(in_features=784, out_features=128, bias=True)
              (1): Linear(in_features=128, out_features=10, bias=True)
            )

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc

        Returns:
            Model: model for a given stage.
        """
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage.

        Example::

            # for typical classification task
            >>> experiment.get_criterion(stage="training")
            nn.CrossEntropyLoss()

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc

        Returns:
            Criterion: criterion for a given stage.
        """
        pass

    @abstractmethod
    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        """Returns the optimizer for a given stage and model.

        Example::

            >>> experiment.get_optimizer(stage="training", model=model)
            torch.optim.Adam(model.parameters())

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc
            model (Model): model to optimize with stage optimizer

        Returns:
            Optimizer: optimizer for a given stage and model.
        """
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        """Returns the scheduler for a given stage and optimizer.

        Example::
            >>> experiment.get_scheduler(stage="training", optimizer=optimizer)
            torch.optim.lr_scheduler.StepLR(optimizer)

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc
            optimizer (Optimizer): optimizer to schedule with stage scheduler

        Returns:
            Scheduler: scheduler for a given stage and optimizer.
        """
        pass

    def get_experiment_components(
        self, model: nn.Module, stage: str
    ) -> Tuple[Criterion, Optimizer, Scheduler]:
        """
        Returns the tuple containing criterion, optimizer and scheduler by
        giving model and stage.

        Aggregation method, based on,

        - :py:mod:`catalyst.core.experiment._Experiment.get_criterion`
        - :py:mod:`catalyst.core.experiment._Experiment.get_optimizer`
        - :py:mod:`catalyst.core.experiment._Experiment.get_scheduler`

        Args:
            model (Model): model to optimize with stage optimizer
            stage (str): stage name of interest,
                like "pretraining" / "training" / "finetuning" / etc

        Returns:
            tuple: criterion, optimizer, scheduler for a given stage and model
        """
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return criterion, optimizer, scheduler

    def get_transforms(self, stage: str = None, dataset: str = None):
        """Returns the data transforms for a given stage and dataset.

        Args:
            stage (str): stage name of interest,
                like "pretraining" / "training" / "finetuning" / etc
            dataset (str): dataset name of interest,
                like "train" / "valid" / "infer"

        .. note::
            For datasets/loaders nameing please follow
            :py:mod:`catalyst.core.state.State` documentation.

        Returns:
            Data transformations to use for specified dataset.

        """
        raise NotImplementedError

    def get_datasets(
        self, stage: str, epoch: int = None, **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        """Returns the datasets for a given stage and epoch.

        .. note::
            For Deep Learning cases you have the same dataset
            during whole stage.

            For Reinforcement Learning it common to change the dataset
            (experiment) every training epoch.

        Args:
            stage (str): stage name of interest,
                like "pretraining" / "training" / "finetuning" / etc
            epoch (int): epoch index
            **kwargs (dict): additional parameters to use during
                dataset creation

        Returns:
            OrderedDict[str, Dataset]: Ordered dictionary
                with datasets for current stage and epoch.

        .. note::
            We need ordered dictionary to guarantee the correct dataflow
            and order of our training datasets.
            For example, to run through train data before validation one :)

        Example::

            >>> experiment.get_datasets(
            >>>     stage="training",
            >>>     in_csv_train="path/to/train/csv",
            >>>     in_csv_valid="path/to/valid/csv",
            >>> )
            OrderedDict({
                "train": CsvDataset(in_csv=in_csv_train, ...),
                "valid": CsvDataset(in_csv=in_csv_valid, ...),
            })


        """
        raise NotImplementedError

    @abstractmethod
    def get_loaders(
        self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage.

        .. note::
            Wrapper for
            :py:mod:`catalyst.core.experiment._Experiment.get_datasets`.
            For most of your experiments you need to rewrite `get_datasets`
            method only.

        Args:
            stage (str): stage name of interest,
                like "pretraining" / "training" / "finetuning" / etc
            epoch (int): epoch index
            **kwargs (dict): additional parameters to use during
                dataset creation

        Returns:
            OrderedDict[str, DataLoader]: Ordered dictionary
                with loaders for current stage and epoch.

        """
        raise NotImplementedError

    @abstractmethod
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns callbacks for a given stage.

        .. note::
            To learn more about Catalyst Callbacks mechanism, please follow
            :py:mod:`catalyst.core.callback.Callback` documentation.

        .. note::
            We need ordered dictionary to guarantee the correct dataflow
            and order of metrics optimization.
            For example, to compute loss before optimization,
            or to compute all the metrics before logging :)

        Args:
            stage (str): stage name of interest
                like "pretraining" / "training" / "finetuning" / etc

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary
            with callbacks for current stage.

        .. note::
            To learn more about Catalyst Core concepts, please check out

                - :py:mod:`catalyst.core.experiment._Experiment`
                - :py:mod:`catalyst.core.runner._Runner`
                - :py:mod:`catalyst.core.state.State`
                - :py:mod:`catalyst.core.callback.Callback`
        """
        pass


class StageBasedExperiment(_Experiment):
    """
    Experiment that provides constant
    datasources during training/inference stage.
    """

    def get_native_batch(
        self, stage: str, loader: Union[str, int] = 0, data_index: int = 0
    ):
        """
        Returns a batch from experiment loader.

        Args:
            stage (str): stage name
            loader (Union[str, int]): loader name or its index,
                default is the first loader
            data_index (int): index in dataset from the loader
        """
        loaders = self.get_loaders(stage)
        if isinstance(loader, str):
            _loader = loaders[loader]
        elif isinstance(loader, int):
            _loader = list(loaders.values())[loader]
        else:
            raise TypeError("Loader parameter must be a string or an integer")

        dataset = _loader.dataset
        collate_fn = _loader.collate_fn

        sample = collate_fn([dataset[data_index]])

        return sample


__all__ = ["_Experiment", "StageBasedExperiment"]
