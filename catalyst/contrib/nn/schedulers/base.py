from typing import List, Optional
from abc import ABC

from torch.optim.lr_scheduler import _LRScheduler

from catalyst.utils import set_optimizer_momentum


class BaseScheduler(_LRScheduler, ABC):
    """Base class for all schedulers with momentum update."""

    def get_momentum(self) -> List[float]:
        """Function that returns the new momentum for optimizer.

        Returns:
            List[float]: calculated momentum for every param groups
        """
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """Make one scheduler step.

        Args:
            epoch (int, optional): current epoch num
        """
        super().step(epoch)
        momentums = self.get_momentum()
        for i, momentum in enumerate(momentums):
            set_optimizer_momentum(self.optimizer, momentum, index=i)


class BatchScheduler(BaseScheduler, ABC):
    """@TODO: Docs. Contribution is welcome."""
