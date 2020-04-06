import torch

from catalyst import utils
from catalyst.contrib.nn.schedulers import BatchScheduler, OneCycleLRWithWarmup
from catalyst.core import Callback, CallbackNode, CallbackOrder, State


class SchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduced_metric: str = None
    ):
        super().__init__(order=CallbackOrder.Scheduler, node=CallbackNode.All)
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduced_metric = reduced_metric

    @staticmethod
    def _scheduler_step(
        scheduler,
        reduced_metric=None,
    ):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(reduced_metric)
            lr = scheduler.optimizer.param_groups[0]["lr"]
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]

        momentum = utils.get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum

    def step_batch(self, state: State):
        lr, momentum = self._scheduler_step(scheduler=self._scheduler)

        if self.scheduler_key is not None:
            state.batch_metrics[f"lr/{self.scheduler_key}"] = lr
            if momentum is not None:
                state.batch_metrics[f"momentum/{self.scheduler_key}"] = \
                    momentum
        else:
            state.batch_metrics["lr"] = lr
            if momentum is not None:
                state.batch_metrics["momentum"] = momentum

    def step_epoch(self, state: State):
        reduced_metric = state.valid_metrics[self.reduced_metric]
        lr, momentum = self._scheduler_step(
            scheduler=self._scheduler, reduced_metric=reduced_metric
        )

        if self.scheduler_key is not None:
            state.epoch_metrics[f"lr/{self.scheduler_key}"] = lr
            if momentum is not None:
                state.epoch_metrics[f"momentum/{self.scheduler_key}"] = \
                    momentum
        else:
            state.epoch_metrics["lr"] = lr
            if momentum is not None:
                state.epoch_metrics["momentum"] = momentum

    def on_stage_start(self, state: State):
        self.reduced_metric = self.reduced_metric or state.main_metric

        scheduler = state.get_attr(
            key="scheduler", inner_key=self.scheduler_key
        )
        assert scheduler is not None
        self._scheduler = scheduler

        if self.mode is None:
            if isinstance(scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if isinstance(scheduler, OneCycleLRWithWarmup) and \
                self.mode == "batch":
            scheduler.reset()
        assert self.mode is not None

    def on_loader_start(self, state: State):
        if state.is_train_loader and \
                isinstance(self._scheduler, OneCycleLRWithWarmup) and \
                self.mode == "batch":
            self._scheduler.recalculate(
                loader_len=state.loader_len, current_step=state.epoch
            )

    def on_batch_end(self, state: State):
        if state.is_train_loader and self.mode == "batch":
            self.step_batch(state=state)

    def on_epoch_end(self, state: State):
        if self.mode == "epoch":
            self.step_epoch(state=state)


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""
    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
        super().__init__(order=CallbackOrder.Scheduler, node=CallbackNode.All)
        self.init_lr = 0
        self.optimizer_key = optimizer_key

    def calc_lr(self):
        return None

    def calc_momentum(self):
        return None

    @staticmethod
    def _update_lr(optimizer, new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    @staticmethod
    def _update_momentum(optimizer, new_momentum):
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum

    def _update_optimizer(self, optimizer):
        new_lr = self.calc_lr()
        if new_lr is not None:
            self._update_lr(optimizer, new_lr)

        new_momentum = self.calc_momentum()
        if new_momentum is not None:
            self._update_momentum(optimizer, new_momentum)
        else:
            new_momentum = utils.get_optimizer_momentum(optimizer)

        return new_lr, new_momentum

    def update_optimizer(self, state: State):
        lr, momentum = self._update_optimizer(optimizer=self._optimizer)

        if self.optimizer_key is not None:
            state.batch_metrics[f"lr_{self.optimizer_key}"] = lr
            state.batch_metrics[f"momentum_{self.optimizer_key}"] = momentum
        else:
            state.batch_metrics["lr"] = lr
            state.batch_metrics["momentum"] = momentum

    def on_stage_start(self, state: State):
        optimizer = state.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        self._optimizer = optimizer
        self.init_lr = optimizer.defaults["lr"]

    def on_loader_start(self, state: State):
        if state.is_train_loader:
            self.update_optimizer(state=state)

    def on_batch_end(self, state: State):
        if state.is_train_loader:
            self.update_optimizer(state=state)


__all__ = ["SchedulerCallback", "LRUpdater"]
