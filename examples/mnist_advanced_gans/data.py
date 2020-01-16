from typing import Iterator, Sized
from collections import defaultdict
import random

import numpy as np

from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
# TODO: refactor


class SameClassBatchSampler(Sampler):
    def __init__(self, data_source: Sized, batch_size: int = 64,
                 drop_last: bool = False, shuffle: bool = False) -> None:
        super().__init__(data_source)
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self._build_class_to_indices_mapping(data_source)

    def _build_class_to_indices_mapping(self, dataset):
        # bold approach to get every target and remember it
        class_id2samples = defaultdict(list)
        for idx in range(len(dataset)):
            _, class_id = dataset[idx]
            assert isinstance(class_id, int)
            class_id2samples[class_id].append(idx)
        self.class_id2samples = class_id2samples

        add_one = 1 if self.drop_last else 0
        self.len = sum((len(v) + add_one) // 2 * 2 for k, v in self.class_id2samples.items())
        self.len = self.len // self.batch_size + (1 if self.len % self.batch_size > 0 else 0)

    def __iter__(self):
        indices_lists = [indices.copy() for indices in self.class_id2samples.values()]
        if self.shuffle:
            for l in indices_lists:
                random.shuffle(l)

        # TODO: remove double meaning of "drop_last" (now it is 1)drop_last in each class_id list and 2) drop_last batch
        first_list = []
        second_list = []
        for curr_class_indices in indices_lists:
            # if two same images will be sampled it would be bad for discriminator as generator may learn to be autoencoder
            assert len(curr_class_indices) > 1, "dataset must have at least 2 unique examples for class for batch sampler to work appropriately"

            n_samples = len(curr_class_indices) // 2
            if not self.drop_last and len(curr_class_indices) % 2 == 1:
                n_samples += 1
            first_list += curr_class_indices[:n_samples]
            second_list += curr_class_indices[-n_samples:]

        indices = np.arange(len(first_list))
        np.random.shuffle(indices)
        batch_size = self.batch_size // 2  # because we concatenate two parts of same length
        batch_indices = [indices[idx * batch_size:(idx + 1) * batch_size] for idx in range(self.len)]

        sample_ids = np.array([
            first_list,
            second_list
        ])
        all_batches = [np.concatenate(sample_ids[:, curr_batch_indices]) for curr_batch_indices in batch_indices]
        return iter(all_batches)

    def __len__(self) -> int:
        return self.len
