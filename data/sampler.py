import numpy as np
from torch.utils.data.sampler import Sampler


class BalanceClassSampler(Sampler):
    def __init__(self, labels, mode="downsampling"):
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

    def __iter__(self):
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert (len(indices) == self.length)
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.length
