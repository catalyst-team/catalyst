import safitty

import torch

from catalyst import utils
from catalyst.contrib.nn.schedulers import BatchScheduler, OneCycleLRWithWarmup
from catalyst.core import _State, Callback, CallbackOrder


class SchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduce_metric: str = "loss"
    ):
        super().__init__(CallbackOrder.Scheduler)
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    def step(self, state: _State):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )

        valid_metric = \
            safitty.get(state.metric_manager.valid_values, self.reduce_metric)
        lr, momentum = self._scheduler_step(
            scheduler=scheduler, valid_metric=valid_metric
        )

        state.set_key(lr, key="lr", inner_key=self.scheduler_key)
        state.set_key(momentum, key="momentum", inner_key=self.scheduler_key)

    def on_stage_start(self, state: _State):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        assert scheduler is not None

        if self.mode is None:
            if isinstance(scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if isinstance(scheduler, OneCycleLRWithWarmup) and \
                self.mode == "batch":
            scheduler.reset()

    def on_loader_start(self, state: _State):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        if state.loader_name.startswith("train") and \
                isinstance(scheduler, OneCycleLRWithWarmup) and \
                self.mode == "batch":
            scheduler.recalculate(
                loader_len=state.loader_len, current_step=state.stage_epoch
            )

    def on_batch_end(self, state: _State):
        if self.mode == "batch":
            self.step(state=state)

    def on_epoch_end(self, state: _State):
        if self.mode == "epoch":
            self.step(state=state)

    @staticmethod
    def _scheduler_step(
        scheduler,
        valid_metric=None,
    ):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_metric)
            lr = safitty.get(scheduler.optimizer.param_groups, 0, "lr")
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]

        momentum = utils.get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""

    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
        super().__init__(CallbackOrder.Scheduler)
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

    def update_optimizer(self, state: _State):
        if not state.need_backward:
            return

        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        lr, momentum = self._update_optimizer(optimizer=optimizer)
        state.set_key(lr, key="lr", inner_key=self.optimizer_key)
        state.set_key(momentum, key="momentum", inner_key=self.optimizer_key)

    def on_stage_start(self, state: _State):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        self.init_lr = optimizer.defaults["lr"]

    def on_loader_start(self, state: _State):
        if state.need_backward:
            self.update_optimizer(state=state)

    def on_batch_end(self, state: _State):
        if state.need_backward:
            self.update_optimizer(state=state)


__all__ = ["SchedulerCallback", "LRUpdater"]
