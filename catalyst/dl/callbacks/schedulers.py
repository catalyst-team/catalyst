from typing import Tuple

from .core import Callback
from catalyst.dl.callbacks.utils import get_optimizer_momentum


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""

    def __init__(self, optimizer_key: str = None):
        """
        :param optimizer_key: which optimizer key to use
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


class OneCycleLR(LRUpdater):
    """
    An learning rate updater
        that implements the Circular Learning Rate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    """

    def __init__(
        self,
        cycle_len: int,
        div_factor: float,
        increase_fraction: float,
        momentum_range: Tuple[float, float],
        optimizer_key: str = None
    ):
        """

        :param init_lr: init learning rate for torch optimizer
        :param cycle_len: (int) num epochs to apply one cycle policy
        :param div_factor: max_lr / init_lr
        :param increase_fraction:
            the part of cycle when learning rate increases
        :param momentum_range: (tuple(int, int)) max and min momentum values
        :param optimizer_key: which optimizer key to use
            for learning rate scheduling
        """
        super().__init__(optimizer_key=optimizer_key)
        self.total_iter = None
        self.cycle_len = cycle_len
        self.momentum_range = momentum_range
        self.div_factor = div_factor
        self.increase_fraction = increase_fraction

        self.cycle_iter = 0
        self.cycle_len = cycle_len
        # point in iterations for starting lr decreasing
        self.cut_point = None

    def on_loader_start(self, state):
        if state.need_backward:
            self.total_iter = state.loader_len * self.cycle_len - 1
            self.cut_point = int(self.total_iter * self.increase_fraction)

        super().on_loader_start(state=state)

    def calc_lr(self):
        # calculate percent for learning rate change
        if self.cycle_iter > self.cut_point:
            percent_curr = (self.cycle_iter - self.cut_point)
            percent_all = (self.total_iter - self.cut_point)
            percent = (1 - percent_curr / percent_all)
        else:
            percent = self.cycle_iter / self.cut_point

        current_mult = (1 + percent * (self.div_factor - 1)) / self.div_factor
        res = self.init_lr * current_mult

        self.cycle_iter += 1
        if self.cycle_iter == self.total_iter:
            self.cycle_iter = 0
        return res

    def calc_momentum(self):
        if self.cycle_iter > self.cut_point:
            now_ = (self.cycle_iter - self.cut_point)
            all_ = (self.total_iter - self.cut_point)
            percent = now_ / all_
        else:
            percent = 1 - self.cycle_iter / self.cut_point
        res = (
            self.momentum_range[1] +
            percent * (self.momentum_range[0] - self.momentum_range[1])
        )
        return res


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

        :param final_lr: final learning rate to try with
        :param scale: learning rate increasing scale ("log" or "linear")
        :param num_steps:  number of batches to try;
            if None - whole loader would be used.
        :param optimizer_key: which optimizer key to use
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
