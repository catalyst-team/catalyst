from typing import Optional, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

import torch

from catalyst.contrib.nn.schedulers import BatchScheduler, OneCycleLRWithWarmup
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.utils.torch import get_optimizer_momentum

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ISchedulerCallback(Callback):
    """Scheduler callback interface, abstraction over scheduler step."""

    pass


class SchedulerCallback(ISchedulerCallback):
    """Callback for wrapping schedulers.

    Notebook API example:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.dl import (
            SupervisedRunner, AccuracyCallback,
            CriterionCallback, SchedulerCallback,
        )

        num_samples, num_features = 10_000, 10
        n_classes = 10
        X = torch.rand(num_samples, num_features)
        y = torch.randint(0, n_classes, [num_samples])
        loader = DataLoader(TensorDataset(X, y), batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        model = torch.nn.Linear(num_features, n_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        runner = SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=5,
            verbose=False,
            main_metric="accuracy03",
            minimize_metric=False,
            callbacks=[
                AccuracyCallback(
                    accuracy_args=[1, 3, 5]
                ),
                SchedulerCallback(reduced_metric="loss")
            ]
        )

    Config API usage example:

    .. code-block:: yaml

        stages:
          ...
          scheduler_params:
            scheduler: MultiStepLR
            milestones: [1]
            gamma: 0.3
          ...
          stage_N:
            ...
            callbacks_params:
              ...
              scheduler:
                callback: SchedulerCallback
                # arguments for SchedulerCallback
                reduced_metric: loss
          ...
    """

    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduced_metric: str = None,
    ):
        """
        Args:
            scheduler_key: scheduler name, if ``None``,
                default is ``None``.
            mode: scheduler mode, should be one of
                ``"epoch"`` or ``"batch"``, default is ``None``.
                If ``None`` and object is instance of ``BatchScheduler``
                or ``OneCycleLRWithWarmup`` then will be used ``"batch"``
                otherwise - ``"epoch"``.
            reduced_metric: metric name to forward to scheduler
                object, if ``None`` then will be used main metric
                specified in experiment.
        """
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduced_metric = reduced_metric

    @staticmethod
    def _scheduler_step(
        scheduler, reduced_metric=None,
    ):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(reduced_metric)
            lr = scheduler.optimizer.param_groups[0]["lr"]
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]

        momentum = get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum

    def step_batch(self, runner: "IRunner") -> None:
        """Update learning rate and momentum in runner.

        Args:
            runner: current runner
        """
        lr, momentum = self._scheduler_step(scheduler=self._scheduler)

        if self.scheduler_key is not None:
            runner.batch_metrics[f"lr/{self.scheduler_key}"] = lr
            if momentum is not None:
                runner.batch_metrics[
                    f"momentum/{self.scheduler_key}"
                ] = momentum
        else:
            runner.batch_metrics["lr"] = lr
            if momentum is not None:
                runner.batch_metrics["momentum"] = momentum

    def step_epoch(self, runner: "IRunner") -> None:
        """Update momentum in runner.

        Args:
            runner: current runner
        """
        reduced_metric = runner.valid_metrics[self.reduced_metric]
        lr, momentum = self._scheduler_step(
            scheduler=self._scheduler, reduced_metric=reduced_metric
        )

        if self.scheduler_key is not None:
            runner.epoch_metrics[f"lr/{self.scheduler_key}"] = lr
            if momentum is not None:
                runner.epoch_metrics[
                    f"momentum/{self.scheduler_key}"
                ] = momentum
        else:
            runner.epoch_metrics["lr"] = lr
            if momentum is not None:
                runner.epoch_metrics["momentum"] = momentum

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook.

        Args:
            runner: current runner
        """
        self.reduced_metric = self.reduced_metric or runner.main_metric

        scheduler = runner.get_attr(
            key="scheduler", inner_key=self.scheduler_key
        )
        assert scheduler is not None
        self._scheduler = scheduler

        if self.mode is None:
            if isinstance(scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if (
            isinstance(scheduler, OneCycleLRWithWarmup)
            and self.mode == "batch"
        ):
            scheduler.reset()
        assert self.mode is not None

    def on_loader_start(self, runner: "IRunner") -> None:
        """Loader start hook.

        Args:
            runner: current runner
        """
        if (
            runner.is_train_loader
            and isinstance(self._scheduler, OneCycleLRWithWarmup)
            and self.mode == "batch"
        ):
            self._scheduler.recalculate(
                loader_len=runner.loader_len, current_step=runner.epoch - 1
            )

    def on_batch_end(self, runner: "IRunner") -> None:
        """Batch end hook.

        Args:
            runner: current runner
        """
        if runner.is_train_loader and self.mode == "batch":
            self.step_batch(runner=runner)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Epoch end hook.

        Args:
            runner: current runner
        """
        if self.mode == "epoch":
            self.step_epoch(runner=runner)


class ILRUpdater(ABC, Callback):
    """Basic class that all Lr updaters inherit from."""

    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self.init_lr = 0
        self.optimizer_key = optimizer_key

    @abstractmethod
    def calc_lr(self):
        """Interface for calculating learning rate."""
        pass

    @abstractmethod
    def calc_momentum(self):
        """Interface for calculating momentum"""
        pass

    @staticmethod
    def _update_lr(optimizer, new_lr) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    @staticmethod
    def _update_momentum(optimizer, new_momentum) -> None:
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum

    def _update_optimizer(self, optimizer) -> Tuple[float, float]:
        new_lr = self.calc_lr()
        if new_lr is not None:
            self._update_lr(optimizer, new_lr)

        new_momentum = self.calc_momentum()
        if new_momentum is not None:
            self._update_momentum(optimizer, new_momentum)
        else:
            new_momentum = get_optimizer_momentum(optimizer)

        return new_lr, new_momentum

    def update_optimizer(self, runner: "IRunner") -> None:
        """Update learning rate and momentum in runner.

        Args:
            runner: current runner
        """
        lr, momentum = self._update_optimizer(optimizer=self._optimizer)

        if self.optimizer_key is not None:
            runner.batch_metrics[f"lr_{self.optimizer_key}"] = lr
            runner.batch_metrics[f"momentum_{self.optimizer_key}"] = momentum
        else:
            runner.batch_metrics["lr"] = lr
            runner.batch_metrics["momentum"] = momentum

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook.

        Args:
            runner: current runner
        """
        optimizer = runner.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        self._optimizer = optimizer
        self.init_lr = optimizer.defaults["lr"]

    def on_loader_start(self, runner: "IRunner") -> None:
        """Loader start hook.

        Args:
            runner: current runner
        """
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        """Batch end hook.

        Args:
            runner: current runner
        """
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)


class LRFinder(ILRUpdater):
    """
    Helps you find an optimal learning rate for a model, as per suggestion of
    `Cyclical Learning Rates for Training Neural Networks`_ paper.
    Learning rate is increased in linear or log scale, depending on user input.

    See `How Do You Find A Good Learning Rate`_ article for details.

    .. _Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186
    .. _How Do You Find A Good Learning Rate:
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(
        self,
        final_lr,
        scale: str = "log",
        num_steps: Optional[int] = None,
        optimizer_key: str = None,
    ):
        """
        Args:
            final_lr: final learning rate to try with
            scale: learning rate increasing scale ("log" or "linear")
            num_steps:  number of batches to try;
                if None - whole loader would be used.
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
        super().__init__(optimizer_key=optimizer_key)

        self.final_lr = final_lr
        self.scale = scale
        self.num_steps = num_steps
        self.multiplier = 0
        self.lr_step = 0
        self.find_iter = 0

        self._calc_lr = None
        if scale == "log":
            self._calc_lr = self._calc_lr_log
        elif scale == "linear":
            self._calc_lr = self._calc_lr_linear
        else:
            raise Exception("Not supported")

    def _calc_lr_log(self):
        return self.init_lr * self.multiplier ** self.find_iter

    def _calc_lr_linear(self):
        return self.init_lr + self.lr_step * self.find_iter

    def calc_lr(self):
        """Calculates learning rate.

        Returns:
            learning rate.
        """
        res = self._calc_lr()
        self.find_iter += 1
        return res

    def calc_momentum(self):
        """Calculates new momentum."""
        pass

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook. Updates scheduler statistics.

        Args:
            runner: current runner
        """
        if runner.is_train_loader:
            lr_step = self.final_lr / self.init_lr
            self.num_steps = self.num_steps or runner.loader_len
            self.multiplier = lr_step ** (1 / self.num_steps)
            self.lr_step = (self.final_lr - self.init_lr) / self.num_steps

        super().on_loader_start(runner=runner)

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook. Make scheduler step and stops iterating if needed.

        Args:
            runner: current runner

        Raises:
            NotImplementedError: at the end of LRFinder
        """
        super().on_batch_end(runner=runner)
        if self.find_iter > self.num_steps:
            raise NotImplementedError("End of LRFinder")


__all__ = ["ISchedulerCallback", "SchedulerCallback", "ILRUpdater", "LRFinder"]
