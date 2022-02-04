from typing import Iterator, List, Optional, Union
from collections import Counter
import logging
import random

import numpy as np

import torch
from torch.utils.data.sampler import BatchSampler, Sampler

from catalyst.contrib.data._misc import find_value_ids

LOGGER = logging.getLogger(__name__)


class BalanceBatchSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and
    classification task.

    .. warning::
        Deprecated realization, used for backward compatibility.
        Please use `BatchBalanceClassSampler` instead.

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

    Args:
        labels: list of classes labeles for each elem in the dataset
        p: number of classes in a batch, should be > 1
        k: number of instances of each class in a batch, should be > 1

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737
    """

    def __init__(self, labels: Union[List[int], np.ndarray], p: int, k: int):
        """Sampler initialisation."""
        super().__init__(self)
        classes = set(labels)

        assert isinstance(p, int) and isinstance(k, int)
        assert (1 < p <= len(classes)) and (1 < k)
        assert all(
            n > 1 for n in Counter(labels).values()
        ), "Each class shoud contain at least 2 instances to fit (1)"

        self._labels = np.array(labels)
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

        for cls_id in random.sample(self._classes, self._num_epoch_classes):
            all_cls_inds = find_value_ids(self._labels, cls_id)

            # we've checked in __init__ that this value must be > 1
            num_samples_exists = len(all_cls_inds)

            if num_samples_exists < self._k:
                selected_inds = random.sample(
                    all_cls_inds, k=num_samples_exists
                ) + random.choices(all_cls_inds, k=self._k - num_samples_exists)
            else:
                selected_inds = random.sample(all_cls_inds, k=self._k)

            inds.extend(selected_inds)

        return iter(inds)


class DynamicBalanceClassSampler(Sampler):
    """
    This kind of sampler can be used for classification tasks with significant
    class imbalance.

    The idea of this sampler that we start with the original class distribution
    and gradually move to uniform class distribution like with downsampling.

    Let's define :math: D_i = #C_i/ #C_min where :math: #C_i is a size of class
    i and :math: #C_min is a size of the rarest class, so :math: D_i define
    class distribution. Also define :math: g(n_epoch) is a exponential
    scheduler. On each epoch current :math: D_i  calculated as
    :math: current D_i  = D_i ^ g(n_epoch),
    after this data samples according this distribution.

    Notes:
         In the end of the training, epochs will contain only
         min_size_class * n_classes examples. So, possible it will not
         necessary to do validation on each epoch. For this reason use
         ControlFlowCallback.

    Examples:

        >>> import torch
        >>> import numpy as np

        >>> from catalyst.data import DynamicBalanceClassSampler
        >>> from torch.utils import data

        >>> features = torch.Tensor(np.random.random((200, 100)))
        >>> labels = np.random.randint(0, 4, size=(200,))
        >>> sampler = DynamicBalanceClassSampler(labels)
        >>> labels = torch.LongTensor(labels)
        >>> dataset = data.TensorDataset(features, labels)
        >>> loader = data.dataloader.DataLoader(dataset, batch_size=8)

        >>> for batch in loader:
        >>>     b_features, b_labels = batch

    Sampler was inspired by https://arxiv.org/abs/1901.06783
    """

    def __init__(
        self,
        labels: List[Union[int, str]],
        exp_lambda: float = 0.9,
        start_epoch: int = 0,
        max_d: Optional[int] = None,
        mode: Union[str, int] = "downsampling",
        ignore_warning: bool = False,
    ):
        """
        Args:
            labels: list of labels for each elem in the dataset
            exp_lambda: exponent figure for schedule
            start_epoch: start epoch number, can be useful for multistage
            experiments
            max_d: if not None, limit on the difference between the most
            frequent and the rarest classes, heuristic
            mode: number of samples per class in the end of training. Must be
            "downsampling" or number. Before change it, make sure that you
            understand how does it work
            ignore_warning: ignore warning about min class size
        """
        assert isinstance(start_epoch, int)
        assert 0 < exp_lambda < 1, "exp_lambda must be in (0, 1)"
        super().__init__(labels)
        self.exp_lambda = exp_lambda
        if max_d is None:
            max_d = np.inf
        self.max_d = max_d
        self.epoch = start_epoch
        labels = np.array(labels)
        samples_per_class = Counter(labels)
        self.min_class_size = min(samples_per_class.values())

        if self.min_class_size < 100 and not ignore_warning:
            LOGGER.warning(
                f"the smallest class contains only"
                f" {self.min_class_size} examples. At the end of"
                f" training, epochs will contain only"
                f" {self.min_class_size * len(samples_per_class)}"
                f" examples"
            )

        self.original_d = {
            key: value / self.min_class_size for key, value in samples_per_class.items()
        }
        self.label2idxes = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, int):
            self.min_class_size = mode
        else:
            assert mode == "downsampling"

        self.labels = labels
        self._update()

    def _update(self) -> None:
        """Update d coefficients."""
        current_d = {
            key: min(value ** self._exp_scheduler(), self.max_d)
            for key, value in self.original_d.items()
        }
        samples_per_classes = {
            key: int(value * self.min_class_size) for key, value in current_d.items()
        }
        self.samples_per_classes = samples_per_classes
        self.length = np.sum(list(samples_per_classes.values()))
        self.epoch += 1

    def _exp_scheduler(self) -> float:
        return self.exp_lambda ** self.epoch

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            iterator of indices of stratified sample
        """
        indices = []
        for key in sorted(self.label2idxes):
            samples_per_class = self.samples_per_classes[key]
            replace_flag = samples_per_class > len(self.label2idxes[key])
            indices += np.random.choice(
                self.label2idxes[key], samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)
        self._update()
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class DynamicLenBatchSampler(BatchSampler):
    """
    A dynamic batch length data sampler.
    Should be used with `catalyst.utils.trim_tensors`.

    Adapted from `Dynamic minibatch trimming to improve BERT training speed`_.

    Args:
        sampler: Base sampler.
        batch_size: Size of minibatch.
        drop_last: If ``True``, the sampler will drop the last batch
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
            "produced an inccorect number of batches."
            f" expected {len(self)}, but yielded {yielded}"
        )


__all__ = [
    "BalanceBatchSampler",
    "DynamicBalanceClassSampler",
]
