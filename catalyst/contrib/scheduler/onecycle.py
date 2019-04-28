import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from catalyst.dl.utils import get_optimizer_momentum, set_optimizer_momentum


class OneCycleLR(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            num_epochs: int,
            lr_range=(1.0, 0.005),
            init_lr: float = None,
            warmup_epochs: int = 0,
            decay_epochs: int = 0,
            momentum_range=(0.8, 0.99, 0.999),
            init_momentum: float = None,
    ):
        if len(lr_range) == 2:
            max_lr, min_lr = lr_range
            final_lr = min_lr
        elif len(lr_range) == 3:
            max_lr, min_lr, final_lr = lr_range

        if len(momentum_range) == 2:
            min_momentum, max_momentum = momentum_range
            final_momentum = max_momentum
        elif len(momentum_range) == 3:
            min_momentum, max_momentum, final_momentum = momentum_range

        if init_lr is None:
            init_lr = optimizer.defaults["lr"]
        if init_momentum is None:
            init_momentum = get_optimizer_momentum(optimizer)

        lr_annealing_epochs = num_epochs - (warmup_epochs + decay_epochs)

        self.final_lr = final_lr

        self.lr_warmup = np.linspace(init_lr, max_lr, warmup_epochs)
        self.lr_annealing = np.linspace(max_lr, min_lr, lr_annealing_epochs)
        self.lr_decay = np.linspace(min_lr, final_lr, decay_epochs)
        self.learning_rates = np.concatenate(
            (self.lr_warmup, self.lr_annealing, self.lr_decay)
        )

        self.momentum_decay = np.linspace(
            init_momentum, min_momentum, warmup_epochs)
        self.momentum_annealing = np.linspace(
            min_momentum, max_momentum, lr_annealing_epochs)
        self.momentum_warmup = np.linspace(
            max_momentum, final_momentum, decay_epochs)
        self.momentums = np.concatenate((
            self.momentum_decay, self.momentum_annealing, self.momentum_warmup
        ))
        super().__init__(optimizer)

    def get_lr(self):
        return [self.learning_rates[self.last_epoch] for _ in self.base_lrs]

    def get_momentum(self):
        return self.momentums[self.last_epoch]

    def step(self, epoch=None):
        super().step(epoch)
        set_optimizer_momentum(self.optimizer, self.get_momentum())
