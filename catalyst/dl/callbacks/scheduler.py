from typing import Optional

from catalyst.core.callbacks import LRUpdater
from catalyst.dl import State


class LRFinder(LRUpdater):
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
            scale (str): learning rate increasing scale ("log" or "linear")
            num_steps (Optional[int]):  number of batches to try;
                if None - whole loader would be used.
            optimizer_key (str): which optimizer key to use
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
        """@TODO: Docs. Contribution is welcome."""
        res = self._calc_lr()
        self.find_iter += 1
        return res

    def on_loader_start(self, state: State):
        """@TODO: Docs. Contribution is welcome.

        Args:
            state (State): current state
        """
        if state.is_train_loader:
            lr_ = self.final_lr / self.init_lr
            self.num_steps = self.num_steps or state.loader_len
            self.multiplier = lr_ ** (1 / self.num_steps)
            self.lr_step = (self.final_lr - self.init_lr) / self.num_steps

        super().on_loader_start(state=state)

    def on_batch_end(self, state: State):
        """@TODO: Docs. Contribution is welcome.

        Args:
            state (State): current state
        """
        super().on_batch_end(state=state)
        if self.find_iter > self.num_steps:
            raise NotImplementedError("End of LRFinder")


__all__ = ["LRFinder"]
