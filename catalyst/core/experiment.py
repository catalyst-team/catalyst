from typing import Any, Dict, Iterable, Mapping
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from catalyst.core.callback import Callback, ICallback
from catalyst.core.engine import IEngine
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.typing import Criterion, Model, Optimizer, Scheduler


class IExperiment(ABC):
    """An abstraction that knows **what** you would like to run.

    IExperiment contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment.IExperiment`
            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.experiments.experiment.Experiment`
        - :py:mod:`catalyst.experiments.config.ConfigExperiment`
        - :py:mod:`catalyst.experiments.supervised.SupervisedExperiment`
    """

    @property
    def seed(self) -> int:
        """
        Experiment's initial seed, used to setup `global seed`
        at the beginning of each stage.
        Additionally, Catalyst Runner setups
        `experiment.initial_seed + runner.global_epoch + 1`
        as `global seed` each epoch.
        Used for experiment reproducibility.

        Example::

            >>> experiment.seed
            42
        """
        return 42

    @property
    def name(self) -> str:
        return "IExperiment"

    @property
    def hparams(self) -> OrderedDict:
        """
        Returns hyper-parameters for current experiment.

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
        return {}

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

    @abstractmethod
    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns extra stage parameters for a given stage.

        Example::

            >>> experiment.get_stage_params(stage="train_2")
            {
                "num_epochs": 42,
                "migrate_model_from_previous_stage": True,
                "migrate_callbacks_from_previous_stage": False,
            }

        Args:
            stage: stage name of interest
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
            >>> experiment.get_model(stage="train")
            Sequential(
             : Linear(in_features=784, out_features=128, bias=True)
             : Linear(in_features=128, out_features=10, bias=True)
            )

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            Model: model for a given stage.
        """
        pass

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage.

        Example::

            # for typical classification task
            >>> experiment.get_criterion(stage="training")
            nn.CrossEntropyLoss()

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc

        Returns:  # noqa: DAR202
            Criterion: criterion for a given stage.
        """
        # @TODO: could we also pass model here for model-criterion interaction?
        return None

    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        """Returns the optimizer for a given stage and model.

        Example::

            >>> experiment.get_optimizer(stage="training", model=model)
            torch.optim.Adam(model.parameters())

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc
            model: model to optimize with stage optimizer

        Returns:  # noqa: DAR202
            Optimizer: optimizer for a given stage and model.
        """
        return None

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        """Returns the scheduler for a given stage and optimizer.

        Example::
            >>> experiment.get_scheduler(stage="training", optimizer=optimizer)
            torch.optim.lr_scheduler.StepLR(optimizer)

        Args:
            stage: stage name of interest
                like "pretrain" / "train" / "finetune" / etc
            optimizer: optimizer to schedule with stage scheduler

        Returns:  # noqa: DAR202
            Scheduler: scheduler for a given stage and optimizer.
        """
        return None

    def get_transforms(
        self, stage: str = None, epoch: int = None, dataset: str = None, **kwargs,
    ):
        """Returns the data transforms for a given stage and dataset.

        Args:
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            dataset: dataset name of interest,
                like "train" / "valid" / "infer"

        .. note::
            For datasets/loaders naming please follow
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

            For Reinforcement Learning it's common to change the dataset
            (experiment) every training epoch.

        Args:
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch: epoch index
            **kwargs: additional parameters to use during
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
    def get_loaders(self, stage: str, epoch: int = None) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage.  # noqa: DAR401

        .. note::
            Wrapper for
            :py:mod:`catalyst.core.experiment.IExperiment.get_datasets`.
            For most of your experiments you need to rewrite `get_datasets`
            method only.

        Args:
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            epoch: epoch index

        Returns:  # noqa: DAR202
            OrderedDict[str, DataLoader]: Ordered dictionary
                with loaders for current stage and epoch.

        """
        pass

    def get_callbacks(self, stage: str) -> "OrderedDict[str, ICallback]":
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
            stage: stage name of interest
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
            stage: stage name of interest,
                like "pretrain" / "train" / "finetune" / etc

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary
                with callbacks for current stage.
        """
        return {}

    def get_engine(self) -> IEngine:
        return None

    def get_trial(self) -> ITrial:
        return None

    def get_loggers(self) -> Dict[str, ILogger]:
        return {}


__all__ = ["IExperiment"]
