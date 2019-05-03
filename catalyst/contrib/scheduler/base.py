from abc import ABC
from typing import Optional, List

from torch.optim.lr_scheduler import _LRScheduler
from catalyst.dl.utils import set_optimizer_momentum


class BaseScheduler(_LRScheduler, ABC):
    """
    Base class for all schedulers with momentum update
    """
    def get_momentum(self) -> List[float]:
        """
        Function that returns the new momentum for optimizer
        Returns:
            List[float]: calculated momentum for every param groups
        """
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Make one scheduler step
        Args:
            epoch (int, optional): current epoch's num
        """
        super().step(epoch)
        momentums = self.get_momentum()
        for i, momentum in enumerate(momentums):
            set_optimizer_momentum(self.optimizer, momentum, index=i)


class BatchScheduler(BaseScheduler):
    """
    Base class for all schedulers with ``step_batch`` function
    """
    def step_batch(
            self,
            total_batches: Optional[int] = None
    ) -> None:
        """
        Make one step on batch
        Args:
            total_batches (int, optional): total count of batches in an epoch
        """
        self.last_epoch -= 1
        self.step()
