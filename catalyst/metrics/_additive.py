from typing import Tuple

import numpy as np

from catalyst.metrics._metric import IMetric


class AdditiveValueMetric(IMetric):
    """This metric computes mean and std values of input data.

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
    """

    def __init__(self, compute_on_call: bool = True):
        """Init AdditiveValueMetric"""
        super().__init__(compute_on_call=compute_on_call)
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def reset(self) -> None:
        """Reset all fields"""
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def update(self, value: float, num_samples: int) -> float:
        """Update mean metric value and std with new value.

        Args:
            value: value to update mean and std with
            num_samples: number of value samples that metrics should be updated with

        Returns:
            last value
        """
        self.value = value
        self.n += 1
        self.num_samples += num_samples

        if self.n == 1:
            # Force a copy in torch/numpy
            self.mean = 0.0 + value  # noqa: WPS345
            self.std = 0.0
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - self.mean_old) * num_samples / float(
                self.num_samples
            )
            self.m_s += (value - self.mean_old) * (value - self.mean) * num_samples
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.num_samples - 1.0))
        return value

    def compute(self) -> Tuple[float, float]:
        """
        Returns mean and std values of all the input data

        Returns:
            tuple of mean and std values
        """
        return self.mean, self.std


__all__ = ["AdditiveValueMetric"]
