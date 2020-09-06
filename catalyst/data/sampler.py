from typing import Iterator, List, Optional, Union
from collections import Counter
from operator import itemgetter
from random import choices, sample

import numpy as np

import torch
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

from catalyst.contrib.utils.misc import find_value_ids
from catalyst.data import DatasetFromSampler


class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.

    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(
        self, labels: List[int], mode: Union[str, int] = "downsampling"
    ):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the dataset
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
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
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class BalanceBatchSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and
    classification task.

    Sampler with the given strategy for the C unique classes dataset:
    - Selection P of C classes for the 1st batch
    - Selection K instances for each class for the 1st batch
    - Selection P of C - P remaining classes for 2nd batch
    - Selection K instances for each class for the 2nd batch
    - ...
    The epoch ends when there are no classes left.
    So, the batch sise is P * K except the last one.

    Thus, in each epoch, all the classes will be selected once, but this
    does not mean that all the instances will be selected during the epoch.

    One of the purposes of this sampler is to be used for
    forming triplets and pos/neg pairs inside the batch.
    To guarante existance of these pairs in the batch,
    P and K should be > 1. (1)

    Behavior in corner cases:
    - If a class does not contain K instances,
    a choice will be made with repetition.
    - If C % P == 1 then one of the classes should be dropped
    otherwise statement (1) will not be met.

    This type of sampling can be found in the classical paper of Person Re-Id,
    where P equals 32 and K equals 4:
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737
    """

    def __init__(self, labels: Union[List[int], np.ndarray], p: int, k: int):
        """
        Args:
            labels: list of classes labeles for each elem in the dataset
            p: number of classes in a batch, should be > 1
            k: number of instances of each class in a batch, should be > 1
        """
        super().__init__(self)
        classes = set(labels)

        assert isinstance(p, int) and isinstance(k, int)
        assert (1 < p <= len(classes)) and (1 < k)
        assert all(
            n > 1 for n in Counter(labels).values()
        ), "Each class shoud contain at least 2 instances to fit (1)"

        self._labels = labels
        self._p = p
        self._k = k

        self._batch_size = self._p * self._k
        self._classes = classes

        # to satisfy statement (1)
        num_classes = len(self._classes)
        if num_classes % self._p == 1:
            self._num_epoch_classes = num_classes - 1
        else:
            self._num_epoch_classes = num_classes

    @property
    def batch_size(self) -> int:
        """
        Returns:
            this value should be used in DataLoader as batch size
        """
        return self._batch_size

    @property
    def batches_in_epoch(self) -> int:
        """
        Returns:
            number of batches in an epoch
        """
        return int(np.ceil(self._num_epoch_classes / self._p))

    def __len__(self) -> int:
        """
        Returns:
            number of samples in an epoch
        """
        return self._num_epoch_classes * self._k

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            indeces for sampling dataset elems during an epoch
        """
        inds = []

        for cls_id in sample(self._classes, self._num_epoch_classes):
            all_cls_inds = find_value_ids(self._labels, cls_id)

            # we've checked in __init__ that this value must be > 1
            num_samples_exists = len(all_cls_inds)

            if num_samples_exists < self._k:
                selected_inds = sample(
                    all_cls_inds, k=num_samples_exists
                ) + choices(all_cls_inds, k=self._k - num_samples_exists)
            else:
                selected_inds = sample(all_cls_inds, k=self._k)

            inds.extend(selected_inds)

        return iter(inds)


class MiniEpochSampler(Sampler):
    """
    Sampler iterates mini epochs from the dataset used by ``mini_epoch_len``.

    Example:
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100, drop_last=True)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100,
        >>>     shuffle="per_epoch")
    """

    def __init__(
        self,
        data_len: int,
        mini_epoch_len: int,
        drop_last: bool = False,
        shuffle: str = None,
    ):
        """
        Args:
            data_len (int): Size of the dataset
            mini_epoch_len (int): Num samples from the dataset used in one
              mini epoch.
            drop_last (bool): If ``True``, sampler will drop the last batches
              if its size would be less than ``batches_per_epoch``
            shuffle (str): one of  ``"always"``, ``"real_epoch"``, or `None``.
              The sampler will shuffle indices
              > "per_mini_epoch" - every mini epoch (every ``__iter__`` call)
              > "per_epoch" -- every real epoch
              > None -- don't shuffle
        """
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
                "Shuffle must be one of ['per_mini_epoch', 'per_epoch']. "
                + f"Got {shuffle}"
            )
        self.shuffle_type = shuffle

    def shuffle(self) -> None:
        """@TODO: Docs. Contribution is welcome."""
        if self.shuffle_type == "per_mini_epoch" or (
            self.shuffle_type == "per_epoch" and self.state_i == 0
        ):
            if self.data_len >= self.mini_epoch_len:
                self.indices = self._indices
                np.random.shuffle(self.indices)
            else:
                self.indices = np.random.choice(
                    self._indices, self.mini_epoch_len, replace=True
                )

    def __iter__(self) -> Iterator[int]:
        """@TODO: Docs. Contribution is welcome."""
        self.state_i = self.state_i % self.divider
        self.shuffle()

        start = self.state_i * self.mini_epoch_len
        stop = (
            self.end_pointer
            if (self.state_i == self.steps)
            else (self.state_i + 1) * self.mini_epoch_len
        )
        indices = self.indices[start:stop].tolist()

        self.state_i += 1
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
            int: length of the mini-epoch
        """
        return self.mini_epoch_len


class DynamicLenBatchSampler(BatchSampler):
    """
    A dynamic batch length data sampler.
    Should be used with `catalyst.utils.trim_tensors`.

    Adapted from `Dynamic minibatch trimming to improve BERT training speed`_.

    Args:
        sampler (torch.utils.data.Sampler): Base sampler.
        batch_size (int): Size of minibatch.
        drop_last (bool): If ``True``, the sampler will drop the last batch
        if its size would be less than ``batch_size``.

    Usage example:

        >>> from torch.utils import data
        >>> from catalyst.data import DynamicLenBatchSampler
        >>> from catalyst import utils

        >>> dataset = data.TensorDataset(
        >>>     input_ids, input_mask, segment_ids, labels
        >>> )

        >>> sampler_ = data.RandomSampler(dataset)
        >>> sampler = DynamicLenBatchSampler(
        >>>     sampler_, batch_size=16, drop_last=False
        >>> )
        >>> loader = data.DataLoader(dataset, batch_sampler=sampler)

        >>> for batch in loader:
        >>>     tensors = utils.trim_tensors(batch)
        >>>     b_input_ids, b_input_mask, b_segment_ids, b_labels = \
        >>>         tuple(t.to(device) for t in tensors)

    .. _`Dynamic minibatch trimming to improve BERT training speed`:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779
    """

    def __iter__(self):
        """
        Iteration over BatchSampler.
        """
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:
                buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx2 for bucket in buckets for idx2 in bucket]

        for idx3 in leftover:
            batch.append(idx3)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, (
            "produced an inccorect number of batches. "
            + "expected %i, but yielded %i" % (len(self), yielded)
        )


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


__all__ = [
    "BalanceClassSampler",
    "BalanceBatchSampler",
    "MiniEpochSampler",
    "DistributedSamplerWrapper",
    "DynamicLenBatchSampler",
]
