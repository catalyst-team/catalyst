from typing import List

import numpy as np
from torch.optim import Optimizer

from catalyst.dl.utils import get_optimizer_momentum
from .base import BaseScheduler


class OneCycleLR(BaseScheduler):
    """
    OneCycle scheduler with warm-up & lr decay stages.
    First stage increases lr from ``init_lr`` to ``max_lr``,
    and called ``warmup``. Also it decreases momentum
    from ``init_momentum`` to ``min_momentum``. Takes ``warmup_epochs`` epochs

    Second is ``annealing`` stage. Decrease lr from ``max_lr`` to ``min_lr``,
    Increase momentum from ``min_momentum`` to ``max_momentum``.

    Third, optional, lr decay.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            num_epochs: int,
            lr_range=(1.0, 0.005),
            init_lr: float = None,
            warmup_epochs: int = 0,
            warmup_fraction: float = None,
            decay_epochs: int = 0,
            decay_fraction: float = None,
            momentum_range=(0.8, 0.99, 0.999),
            init_momentum: float = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_epochs (int): total number of epochs
            lr_range: tuple with two or three elements
                (max_lr, min_lr, [final_lr])
            init_lr (float, optional): initial lr
            warmup_epochs (int): count of epochs for warm-up stage
            warmup_fraction (float, optional): fraction in [0; 1) to calculate
                number of warmup epochs.
                Cannot be set together with ``warmup_epochs``
            decay_epochs (int): count of epochs for lr decay stage
            decay_fraction (float, optional): fraction in [0; 1) to calculate
                number of decay epochs.
                Cannot be set together with ``decay_epochs``
            momentum_range: tuple with two or three elements
                (min_momentum, max_momentum, [final_momentum])
            init_momentum (float, optional): initial momentum
        """
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

        if warmup_fraction is not None:
            assert 0.0 <= warmup_fraction < 1.0 and warmup_epochs == 0, \
                "You should pass either warmup_epochs or " \
                "warmup_fraction in range [0; 1) "
            warmup_epochs = int(num_epochs * warmup_fraction)

        if decay_fraction is not None:
            assert 0.0 <= decay_fraction < 1.0 and decay_epochs == 0, \
                "You should pass either decay_epochs or " \
                "decay_fraction in range [0; 1) "
            decay_epochs = int(num_epochs * decay_fraction)

        lr_annealing_epochs = num_epochs - (warmup_epochs + decay_epochs)

        lr_warmup = np.linspace(init_lr, max_lr, warmup_epochs)
        lr_annealing = np.linspace(max_lr, min_lr, lr_annealing_epochs)
        lr_decay = np.linspace(min_lr, final_lr, decay_epochs)
        self.learning_rates = np.concatenate(
            (lr_warmup, lr_annealing, lr_decay)
        )

        momentum_decay = np.linspace(
            init_momentum, min_momentum, warmup_epochs)
        momentum_annealing = np.linspace(
            min_momentum, max_momentum, lr_annealing_epochs)
        momentum_warmup = np.linspace(
            max_momentum, final_momentum, decay_epochs)
        self.momentums = np.concatenate((
            momentum_decay, momentum_annealing, momentum_warmup
        ))

        self.total_groups = len(optimizer.param_groups)
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Function that returns the new lr for optimizer
        Returns:
            List[float]: calculated lr for every param groups
        """
        lr = self.learning_rates[self.last_epoch]
        return [lr] * self.total_groups

    def get_momentum(self) -> List[float]:
        """
        Function that returns the new momentum for optimizer
        Returns:
            List[float]: calculated momentum for every param groups
        """
        momentum = self.momentums[self.last_epoch]
        return [momentum] * self.total_groups
