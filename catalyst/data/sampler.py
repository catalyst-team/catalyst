from typing import List, Iterator
import numpy as np

from torch.utils.data.sampler import Sampler


class BalanceClassSampler(Sampler):
    """
    Abstraction over data sampler. Allows you to create stratified sample
    on unbalanced classes.
    """

    def __init__(self, labels: List[int], mode: str = "downsampling"):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the datasety
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum()
            for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode \
                if isinstance(mode, int) \
                else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert (len(indices) == self.length)
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class MiniEpochSampler(Sampler):
    r"""Sampler iterates mini epochs from the dataset
        used by ``mini_epoch_len``

    Args:
        data_len (int): Size of the dataset
        mini_epoch_len (int): Num samples from the dataset used in one
            mini epoch.
        drop_last (bool): If ``True``, sampler will drop the last batches if
            its size would be less than ``batches_per_epoch``
        shuffle (str): one of  ``["always", "real_epoch", None]``.
            The sampler will shuffle indices
            > "per_mini_epoch" -- every mini epoch (every ``__iter__`` call)
            > "per_epoch" -- every real epoch
            > None -- don't shuffle

    Example:
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100, \
            drop_last=True)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100, \
            shuffle="per_epoch")
    """

    def __init__(
        self,
        data_len: int,
        mini_epoch_len: int,
        drop_last: bool = False,
        shuffle: str = None
    ):
        super().__init__(None)

        self.data_len = int(data_len)
        self.mini_epoch_len = int(mini_epoch_len)

        self.steps = int(data_len / self.mini_epoch_len)
        self.state_i = 0

        has_reminder = data_len - self.steps * mini_epoch_len > 0
        if self.steps == 0:
            self.divider = 1
        elif has_reminder and not drop_last:
            self.divider = self.steps + 1
        else:
            self.divider = self.steps

        self._indices = np.arange(self.data_len)
        self.indices = self._indices
        self.end_pointer = max(self.data_len, self.mini_epoch_len)

        if not (shuffle is None or shuffle in ["per_mini_epoch", "per_epoch"]):
            raise ValueError(
                f"Shuffle must be one of ['per_mini_epoch', 'per_epoch']. "
                f"Got {shuffle}"
            )
        self.shuffle_type = shuffle

    def shuffle(self):
        if self.shuffle_type == "per_mini_epoch" or \
                (self.shuffle_type == "per_epoch" and self.state_i == 0):
            if self.data_len >= self.mini_epoch_len:
                self.indices = self._indices
                np.random.shuffle(self.indices)
            else:
                self.indices = np.random.choice(
                    self._indices, self.mini_epoch_len, replace=True
                )

    def __iter__(self) -> Iterator[int]:
        self.state_i = self.state_i % self.divider
        self.shuffle()

        start = self.state_i * self.mini_epoch_len
        stop = self.end_pointer \
            if (self.state_i == self.steps) \
            else (self.state_i + 1) * self.mini_epoch_len
        indices = self.indices[start:stop].tolist()

        self.state_i += 1
        return iter(indices)

    def __len__(self) -> int:
        return self.mini_epoch_len


__all__ = ["BalanceClassSampler", "MiniEpochSampler"]
