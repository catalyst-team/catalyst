from typing import Tuple

import torch

from catalyst.contrib.nn.schedulers import BatchScheduler, OneCycleLRWithWarmup
from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class SchedulerCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduced_metric: str = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(order=CallbackOrder.Scheduler, node=CallbackNode.All)
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

        momentum = utils.get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum

    def step_batch(self, runner: IRunner) -> None:
        """@TODO: Docs. Contribution is welcome.

        Args:
            runner (IRunner): current runner
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

    def step_epoch(self, runner: IRunner) -> None:
        """@TODO: Docs. Contribution is welcome.

        Args:
            runner (IRunner): current runner
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

    def on_stage_start(self, runner: IRunner) -> None:
        """Stage start hook.

        Args:
            runner (IRunner): current runner
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

    def on_loader_start(self, runner: IRunner) -> None:
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        if (
            runner.is_train_loader
            and isinstance(self._scheduler, OneCycleLRWithWarmup)
            and self.mode == "batch"
        ):
            self._scheduler.recalculate(
                loader_len=runner.loader_len, current_step=runner.epoch
            )

    def on_batch_end(self, runner: IRunner) -> None:
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        if runner.is_train_loader and self.mode == "batch":
            self.step_batch(runner=runner)

    def on_epoch_end(self, runner: IRunner) -> None:
        """Epoch end hook.

        Args:
            runner (IRunner): current runner
        """
        if self.mode == "epoch":
            self.step_epoch(runner=runner)


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from."""

    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key (str): which optimizer key to use
                for learning rate scheduling
        """
        super().__init__(order=CallbackOrder.Scheduler, node=CallbackNode.All)
        self.init_lr = 0
        self.optimizer_key = optimizer_key

    def calc_lr(self):
        """@TODO: Docs. Contribution is welcome."""
        return None

    def calc_momentum(self):
        """@TODO: Docs. Contribution is welcome."""
        return None

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
            new_momentum = utils.get_optimizer_momentum(optimizer)

        return new_lr, new_momentum

    def update_optimizer(self, runner: IRunner) -> None:
        """@TODO: Docs. Contribution is welcome.

        Args:
            runner (IRunner): current runner
        """
        lr, momentum = self._update_optimizer(optimizer=self._optimizer)

        if self.optimizer_key is not None:
            runner.batch_metrics[f"lr_{self.optimizer_key}"] = lr
            runner.batch_metrics[f"momentum_{self.optimizer_key}"] = momentum
        else:
            runner.batch_metrics["lr"] = lr
            runner.batch_metrics["momentum"] = momentum

    def on_stage_start(self, runner: IRunner) -> None:
        """Stage start hook.

        Args:
            runner (IRunner): current runner
        """
        optimizer = runner.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        self._optimizer = optimizer
        self.init_lr = optimizer.defaults["lr"]

    def on_loader_start(self, runner: IRunner) -> None:
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)

    def on_batch_end(self, runner: IRunner) -> None:
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)


__all__ = ["SchedulerCallback", "LRUpdater"]
