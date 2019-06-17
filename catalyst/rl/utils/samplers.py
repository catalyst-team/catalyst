import numpy as np
from torch.utils.data import Sampler


class OffpolicyReplaySampler(Sampler):
    def __init__(self, buffer, epoch_len, batch_size):
        super().__init__(None)
        self.buffer = buffer
        self.buffer_history_len = buffer.history_len
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.len = self.epoch_len * self.batch_size

    def __iter__(self):
        indices = np.random.choice(range(len(self.buffer)), size=self.len)
        return iter(indices)

    def __len__(self):
        return self.len


class OnpolicyRolloutSampler(Sampler):
    def __init__(self, buffer, num_mini_epochs):
        super().__init__(None)
        self.buffer = buffer
        self.num_mini_epochs = num_mini_epochs
        buffer_len = len(self.buffer)
        self.len = buffer_len * num_mini_epochs

        indices = []
        for i in range(num_mini_epochs):
            idx = np.arange(buffer_len)
            np.random.shuffle(idx)
            indices.append(idx)
        self.indices = np.concatenate(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.len
