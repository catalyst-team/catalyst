"""
MSE and RMSE meters.
"""
import math

import torch

from . import meter


class MSEMeter(meter.Meter):
    """
    This meter can handle MSE and RMSE.
    Root calculation can be toggled(not calculated by default).
    """

    def __init__(self, root: bool = False):
        """
        Args:
            root (bool): Toggle between calculation of
                RMSE (True) and MSE (False)
        """
        super(MSEMeter, self).__init__()
        self.reset()
        self.root = root

    def reset(self) -> None:
        """Reset meter number of elements and squared error sum."""
        self.n = 0
        self.sesum = 0.0

    def add(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Update squared error stored sum and number of elements.

        Args:
            output (Tensor): Model output tensor or numpy array
            target (Tensor): Target tensor or numpy array
        """
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        self.n += output.numel()
        self.sesum += torch.sum((output - target) ** 2)

    def value(self) -> float:
        """Calculate MSE and return RMSE or MSE.

        Returns:
            float: Root of MSE if `self.root` is True else MSE
        """
        mse = self.sesum / max(1, self.n)
        return math.sqrt(mse) if self.root else mse
