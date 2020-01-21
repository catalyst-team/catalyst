"""
Utility functions and classes for working with data
(datasets, loaders, samplers)
"""
import random
from collections import defaultdict
from typing import Sized

import numpy as np
from torch.utils.data import Sampler


class SameClassBatchSampler(Sampler):
    """
    BatchSampler which samples same classes in first and second batch halves
    (i.e. batch[target][:batch_size//2] == batch[target][batch_size//2:])
    """

    def __init__(self,
                 data_source: Sized,
                 batch_size: int = 64,
                 drop_odd_class_elements: bool = False,
                 drop_last_batch: bool = False,
                 shuffle: bool = False):
        """

        :param data_source: torch dataset which returns (data, class_id)
        :param batch_size: generated batch size
        :param drop_odd_class_elements:
            for each class with odd number of examples, one example
            (in the middle) will be dropped if this value is set to True.
            if set to False, the example in the middle will be repeated twice
            (for shuffled data order is changed every time, so this repeated
            object will not be the same)
        :param drop_last_batch: if True, drops last incomplete batch
        :param shuffle: shuffle walking order
        """
        super().__init__(data_source)
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.drop_odd_class_elements = drop_odd_class_elements
        self.drop_last_batch = drop_last_batch
        self.shuffle = shuffle

        self._build_class_to_indices_mapping(data_source)

    def _build_class_to_indices_mapping(self, dataset):
        # map each class to examples
        self.class_id2samples = defaultdict(list)
        # bold approach to get every target and remember it
        # if the dataset is large this may consume significant amount of time
        for idx in range(len(dataset)):
            _, class_id = dataset[idx]
            assert isinstance(class_id, int)
            self.class_id2samples[class_id].append(idx)

        # compute len (number of batches)
        add_one = 1 if self.drop_odd_class_elements else 0
        self.len = sum((len(v) + add_one) // 2 * 2 for k, v in
                       self.class_id2samples.items())
        add_one = 0
        if self.len % self.batch_size > 0 and not self.drop_last_batch:
            add_one = 1
        self.len = self.len // self.batch_size + add_one

    def __iter__(self):
        # TODO:
        #  check if it does not fail with multiple workers
        #  (I can't right now: windows =( )
        all_batches = self._get_batch_indices()
        return iter(all_batches)

    def __len__(self) -> int:
        return self.len

    def _get_batch_indices(self):
        indices_lists = [indices.copy() for indices in
                         self.class_id2samples.values()]
        if self.shuffle:
            for l in indices_lists:
                random.shuffle(l)

        first_halves = []
        second_halves = []
        for curr_class_indices in indices_lists:
            # if two same images will be sampled it would be bad
            # for discriminator as generator may learn to be autoencoder
            assert len(curr_class_indices) > 1, \
                "dataset must have at least 2 unique examples for class " \
                "for SameClassBatchSampler to work appropriately"

            n_samples = len(curr_class_indices) // 2
            if (
                    not self.drop_odd_class_elements
                    and len(curr_class_indices) % 2 == 1
            ):
                n_samples += 1
            first_halves += curr_class_indices[:n_samples]
            second_halves += curr_class_indices[-n_samples:]

        indices = np.arange(len(first_halves))
        np.random.shuffle(indices)
        batch_size = self.batch_size // 2  # as we concat two parts of same len
        batch_indices = [
            indices[idx * batch_size:(idx + 1) * batch_size]
            for idx in range(self.len)
        ]

        sample_ids = np.array([
            first_halves,
            second_halves
        ])
        all_batches = [
            np.concatenate(sample_ids[:, curr_batch_indices])
            for curr_batch_indices in batch_indices
        ]
        return all_batches
