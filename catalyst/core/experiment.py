from typing import Any, Dict, Iterable, Mapping, Tuple
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader, Dataset

from catalyst.core.callback import Callback
from catalyst.tools.typing import Criterion, Model, Optimizer, Scheduler


class IExperiment(ABC):
    """
    An abstraction that contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.
    In general, the Experiment knows **what** you would like to run.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment.IExperiment`
            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.dl.experiment.base.BaseExperiment`
        - :py:mod:`catalyst.dl.experiment.config.ConfigExperiment`
        - :py:mod:`catalyst.dl.experiment.supervised.SupervisedExperiment`
    """

    @property
    @abstractmethod
    def initial_seed(self) -> int:
        """
        Experiment's initial seed, used to setup `global seed`
        at the beginning of each stage.
        Additionally, Catalyst Runner setups
        `experiment.initial_seed + runner.global_epoch + 1`
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
    def hparams(self) -> OrderedDict:
        """Returns hyper-parameters

        Example::
            >>> experiment.hparams
            OrderedDict([('optimizer', 'Adam'),
             ('lr', 0.02),
             ('betas', (0.9, 0.999)),
             ('eps', 1e-08),
             ('weight_decay', 0),
             ('amsgrad', False),
             ('train_batch_size', 32)])
        """

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

    @abstractmethod
    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns extra stage parameters for a given stage.

        Example::

            >>> experiment.get_stage_params(stage="training")
            {
                "logdir": "./logs/training",
                "num_epochs": 42,
                "valid_loader": "valid",
                "main_metric": "loss",
                "minimize_metric": True,
                "checkpoint_data": {
                    "comment": "break the cycle - use the Catalyst"
                }
            }

        Args:
            stage (str): stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            dict: parameters for a given stage.
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
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
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
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
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
                like "pretrain" / "train" / "finetune" / etc
            model (Model): model to optimize with stage optimizer

        Returns:  # noqa: DAR202
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
                like "pretrain" / "train" / "finetune" / etc
            optimizer (Optimizer): optimizer to schedule with stage scheduler

        Returns:  # noqa: DAR202
            Scheduler: scheduler for a given stage and optimizer.
        """
        pass

    def get_experiment_components(
        self, stage: str, model: nn.Module = None,
    ) -> Tuple[Model, Criterion, Optimizer, Scheduler]:
        """
        Returns the tuple containing criterion, optimizer and scheduler by
        giving model and stage.

        Aggregation method, based on,

        - :py:mod:`catalyst.core.experiment.IExperiment.get_model`
        - :py:mod:`catalyst.core.experiment.IExperiment.get_criterion`
        - :py:mod:`catalyst.core.experiment.IExperiment.get_optimizer`
        - :py:mod:`catalyst.core.experiment.IExperiment.get_scheduler`

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            model (Model): model to optimize with stage optimizer

        Returns:
            tuple: model, criterion, optimizer, scheduler
                for a given stage and model
        """
        if model is None:
            model = self.get_model(stage)
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return model, criterion, optimizer, scheduler

    def get_transforms(self, stage: str = None, dataset: str = None):
        """Returns the data transforms for a given stage and dataset.

        # noqa: DAR401, W505

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            dataset (str): dataset name of interest,
                like "train" / "valid" / "infer"

        .. note::
            For datasets/loaders nameing please follow
            :py:mod:`catalyst.core.runner` documentation.

        Returns:  # noqa: DAR202
            Data transformations to use for specified dataset.

        """
        raise NotImplementedError

    def get_datasets(
        self, stage: str, epoch: int = None, **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        """Returns the datasets for a given stage and epoch.  # noqa: DAR401

        .. note::
            For Deep Learning cases you have the same dataset
            during whole stage.

            For Reinforcement Learning it common to change the dataset
            (experiment) every training epoch.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch (int): epoch index
            **kwargs (dict): additional parameters to use during
                dataset creation

        Returns:  # noqa: DAR202
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
        """Returns the loaders for a given stage.  # noqa: DAR401

        .. note::
            Wrapper for
            :py:mod:`catalyst.core.experiment.IExperiment.get_datasets`.
            For most of your experiments you need to rewrite `get_datasets`
            method only.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch (int): epoch index

        Returns:  # noqa: DAR202
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
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            OrderedDict[str, Callback]: Ordered dictionary  # noqa: DAR202
            with callbacks for current stage.

        .. note::
            To learn more about Catalyst Core concepts, please check out

                - :py:mod:`catalyst.core.experiment.IExperiment`
                - :py:mod:`catalyst.core.runner.IRunner`
                - :py:mod:`catalyst.core.callback.Callback`

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary
                with callbacks for current stage.
        """
        pass


__all__ = ["IExperiment"]
