from typing import List

import numpy as np
from torch.optim import Optimizer

from catalyst.contrib.nn.schedulers.base import BatchScheduler
from catalyst.utils.torch import get_optimizer_momentum


class OneCycleLRWithWarmup(BatchScheduler):
    """OneCycle scheduler with warm-up & lr decay stages.

    First stage increases lr from ``init_lr`` to ``max_lr``,
    and called ``warmup``. Also it decreases momentum
    from ``init_momentum`` to ``min_momentum``. Takes ``warmup_steps`` steps

    Second is ``annealing`` stage. Decrease lr from ``max_lr`` to ``min_lr``,
    Increase momentum from ``min_momentum`` to ``max_momentum``.

    Third, optional, lr decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        lr_range=(1.0, 0.005),
        init_lr: float = None,
        warmup_steps: int = 0,
        warmup_fraction: float = None,
        decay_steps: int = 0,
        decay_fraction: float = None,
        momentum_range=(0.8, 0.99, 0.999),
        init_momentum: float = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_steps: total number of steps
            lr_range: tuple with two or three elements
                (max_lr, min_lr, [final_lr])
            init_lr (float, optional): initial lr
            warmup_steps: count of steps for warm-up stage
            warmup_fraction (float, optional): fraction in [0; 1) to calculate
                number of warmup steps.
                Cannot be set together with ``warmup_steps``
            decay_steps: count of steps for lr decay stage
            decay_fraction (float, optional): fraction in [0; 1) to calculate
                number of decay steps.
                Cannot be set together with ``decay_steps``
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

        warmup_steps = self._calculate_warmup(num_steps, warmup_steps, warmup_fraction)

        decay_steps = self._calculate_decay(num_steps, decay_steps, decay_fraction)

        lr_annealing_steps = num_steps - (warmup_steps + decay_steps)

        self.warmup_steps = warmup_steps
        self.lr_annealing_steps = lr_annealing_steps
        self.decay_steps = decay_steps
        self.num_steps = warmup_steps + lr_annealing_steps + decay_steps

        self.lr_range = init_lr, max_lr, min_lr, final_lr
        self.momentum_range = (
            init_momentum,
            min_momentum,
            max_momentum,
            final_momentum,
        )

        self._calculate_lr_momentum(warmup_steps, lr_annealing_steps, decay_steps)

        self.total_groups = len(optimizer.param_groups)
        super().__init__(optimizer)

    def _calculate_warmup(self, num_steps: int, warmup_steps: int, warmup_fraction: float):
        if warmup_fraction is not None:
            assert 0.0 <= warmup_fraction < 1.0 and warmup_steps == 0, (
                "You should pass either warmup_steps or " "warmup_fraction in range [0; 1) "
            )
            warmup_steps = int(num_steps * warmup_fraction)

        self.warmup_steps = warmup_steps
        self.has_warmup = warmup_steps != 0
        return self.warmup_steps

    def _calculate_decay(self, num_steps: int, decay_steps: int, decay_fraction: float):
        if decay_fraction is not None:
            assert 0.0 <= decay_fraction < 1.0 and decay_steps == 0, (
                "You should pass either decay_steps or " "decay_fraction in range [0; 1) "
            )
            decay_steps = int(num_steps * decay_fraction)

        self.decay_steps = decay_steps
        self.has_decay = decay_steps != 0
        return self.decay_steps

    def _calculate_lr_momentum(self, warmup_steps: int, lr_annealing_steps: int, decay_steps: int):
        init_lr, max_lr, min_lr, final_lr = self.lr_range
        init_momentum, min_momentum, max_momentum, final_momentum = self.momentum_range

        lr_warmup = np.linspace(init_lr, max_lr, warmup_steps)
        lr_annealing = np.linspace(max_lr, min_lr, lr_annealing_steps)
        lr_decay = np.linspace(min_lr, final_lr, decay_steps)

        self.learning_rates = np.concatenate((lr_warmup, lr_annealing, lr_decay))

        momentum_decay = np.linspace(init_momentum, min_momentum, warmup_steps)
        momentum_annealing = np.linspace(min_momentum, max_momentum, lr_annealing_steps)
        momentum_warmup = np.linspace(max_momentum, final_momentum, decay_steps)

        self.momentums = np.concatenate((momentum_decay, momentum_annealing, momentum_warmup))

    def _get_steps_lr_momentum(self, step_num: int):
        if step_num < len(self.learning_rates):
            lr = self.learning_rates[step_num]
        else:
            _, _, _, final_lr = self.lr_range
            lr = final_lr

        if step_num < len(self.momentums):
            momentum = self.momentums[step_num]
        else:
            _, _, _, final_momentum = self.momentum_range
            momentum = final_momentum
        return lr, momentum

    def get_lr(self) -> List[float]:
        """Function that returns the new lr for optimizer.

        Returns:
            List[float]: calculated lr for every param groups
        """
        lr, _ = self._get_steps_lr_momentum(self.last_epoch)
        return [lr] * self.total_groups

    def get_momentum(self) -> List[float]:
        """Function that returns the new momentum for optimizer.

        Returns:
            List[float]: calculated momentum for every param groups
        """
        _, momentum = self._get_steps_lr_momentum(self.last_epoch)
        return [momentum] * self.total_groups

    def reset(self):
        """@TODO: Docs. Contribution is welcome."""
        self._calculate_lr_momentum(self.warmup_steps, self.lr_annealing_steps, self.decay_steps)
        self.last_epoch = 0

    def recalculate(self, loader_batch_len: int, current_batch_step: int) -> None:
        """Recalculates total num_steps for ``batch`` mode.

        Args:
            loader_batch_len: total count of batches in an epoch
            current_batch_step: current step
        """
        warmup_steps = self.warmup_steps * loader_batch_len
        lr_annealing_steps = self.lr_annealing_steps * loader_batch_len
        decay_steps = self.decay_steps * loader_batch_len

        self._calculate_lr_momentum(warmup_steps, lr_annealing_steps, decay_steps)
        self.last_epoch = current_batch_step * loader_batch_len


__all__ = ["OneCycleLRWithWarmup"]
