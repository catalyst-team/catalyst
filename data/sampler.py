import random
import numpy as np

from torch.utils.data.sampler import Sampler


class BalanceClassSampler(Sampler):
    def __init__(self, labels, mode="downsampling"):
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum()
            for label in set(labels)}

        # @TODO: speedup this
        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)}

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode \
                if isinstance(mode, int) \
                else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self):
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key],
                self.samples_per_class,
                replace=replace_).tolist()
        assert (len(indices) == self.length)
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.length


class TripletSampler(Sampler):
    # @TODO: test
    def __init__(self, labels, n_pos=2, n_neg=1, replace=False):
        super().__init__(labels)

        labels = np.array(labels)

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.replace = replace
        self.labels = labels
        # @TODO: speedup this
        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)}
        self.length = len(self.lbl2idx) * (self.n_pos + self.n_neg)

    def __iter__(self):
        indices = []
        keys = sorted(self.lbl2idx)
        random.shuffle(keys)
        for anchor_key in keys:
            # anchor_idx = np.random.choice(
            #     self.lbl2idx[anchor_key],
            #     1).tolist()[0]
            pos_idx = np.random.choice(
                self.lbl2idx[anchor_key],  # .copy().pop(anchor_idx),
                self.n_pos, replace=self.replace).tolist()
            other_keys = list(self.lbl2idx.keys())
            other_keys.remove(anchor_key)
            neg_key = random.choice(other_keys)
            neg_idx = np.random.choice(
                self.lbl2idx[neg_key],
                self.n_neg, replace=self.replace).tolist()
            # indices += [anchor_idx] + pos_idx + neg_idx
            indices += pos_idx + neg_idx
        assert (len(indices) == self.length)
        return iter(indices)

    def __len__(self):
        return self.length
