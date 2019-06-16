import safitty
import torch

from catalyst.dl.core import Callback, RunnerState
from catalyst.contrib.schedulers import OneCycleLR, BatchScheduler
from catalyst.utils import get_optimizer_momentum


class SchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduce_metric: str = "loss"
    ):
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    def step(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )

        valid_metric = \
            safitty.get(state.metrics.valid_values, self.reduce_metric)
        lr, momentum = self._scheduler_step(
            scheduler=scheduler,
            valid_metric=valid_metric
        )

        state.set_key(lr, key="lr", inner_key=self.scheduler_key)
        state.set_key(momentum, key="momentum", inner_key=self.scheduler_key)

    def on_stage_start(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        assert scheduler is not None

        if self.mode is None:
            if isinstance(scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if isinstance(scheduler, OneCycleLR) and self.mode == "batch":
            scheduler.reset()

    def on_loader_start(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        if state.loader_name.startswith("train") and \
                isinstance(scheduler, OneCycleLR) and self.mode == "batch":
            scheduler.recalculate(
                loader_len=state.loader_len,
                current_step=state.stage_epoch
            )

    def on_batch_end(self, state):
        if self.mode == "batch":
            self.step(state=state)

    def on_epoch_end(self, state):
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

        momentum = get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""

    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
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
            new_momentum = get_optimizer_momentum(optimizer)

        return new_lr, new_momentum

    def update_optimizer(self, state):
        if not state.need_backward:
            return

        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        lr, momentum = self._update_optimizer(optimizer=optimizer)
        state.set_key(lr, key="lr", inner_key=self.optimizer_key)
        state.set_key(momentum, key="momentum", inner_key=self.optimizer_key)

    def on_stage_start(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        self.init_lr = optimizer.defaults["lr"]

    def on_loader_start(self, state):
        if state.need_backward:
            self.update_optimizer(state=state)

    def on_batch_end(self, state):
        if state.need_backward:
            self.update_optimizer(state=state)


class LRFinder(LRUpdater):
    """
    Helps you find an optimal learning rate for a model,
    as per suggestion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(
        self,
        final_lr,
        scale="log",
        num_steps=None,
        optimizer_key=None
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

    def calc_lr(self):
        res = self._calc_lr()
        self.find_iter += 1
        return res

    def _calc_lr_log(self):
        return self.init_lr * self.multiplier**self.find_iter

    def _calc_lr_linear(self):
        return self.init_lr + self.lr_step * self.find_iter

    def on_loader_start(self, state):
        if state.need_backward:
            lr_ = self.final_lr / self.init_lr
            self.num_steps = self.num_steps or state.loader_len
            self.multiplier = lr_**(1 / self.num_steps)
            self.lr_step = (self.final_lr - self.init_lr) / self.num_steps

        super().on_loader_start(state=state)

    def on_batch_end(self, state):
        super().on_batch_end(state=state)
        if self.find_iter > self.num_steps:
            raise NotImplementedError("End of LRFinder")


__all__ = ["SchedulerCallback", "LRUpdater", "LRFinder"]
