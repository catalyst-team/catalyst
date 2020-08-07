"""
Average value meter
"""
import numpy as np

from catalyst.tools.meters import meter


class AverageValueMeter(meter.Meter):
    """
    Average value meter stores mean and standard deviation
    for population of input values.
    Meter updates are applied online, one value for each update.
    Values are not cached, only the last added.
    """

    def __init__(self):
        """Constructor method for the ``AverageValueMeter`` class."""
        super(AverageValueMeter, self).__init__()
        self.n = 0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.n_samples = 0

    def add(self, value, batch_size) -> None:
        """Add a new observation.

        Updates of mean and std are going online, with
        `Welford's online algorithm
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

        Args:
            value (float): value for update,
                can be scalar number or PyTorch tensor
            batch_size (int): batch size for update

        .. note::
            Because of algorithm design,
            you can update meter values with only one value a time.
        """
        self.val = value
        self.n += 1
        self.n_samples += batch_size

        if self.n == 1:
            # Force a copy in torch/numpy
            self.mean = 0.0 + value  # noqa: WPS345
            self.std = 0.0
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (
                value - self.mean_old
            ) * batch_size / float(self.n_samples)
            self.m_s += (
                (value - self.mean_old) * (value - self.mean) * batch_size
            )
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n_samples - 1.0))

    def value(self):
        """Returns meter values.

        Returns:
            Tuple[float, float]: tuple of mean and std
            that have been updated online.
        """
        return self.mean, self.std

    def reset(self):
        """Resets the meter to default settings."""
        self.n = 0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.n_samples = 0


__all__ = ["AverageValueMeter"]
