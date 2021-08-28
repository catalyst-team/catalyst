import numpy as np
from torch.utils.data import Dataset, IterableDataset


class simCLRDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform_left, transform_right=None):
        super().__init__()
        self.transform_left = transform_left
        if not isinstance(transform_right, None):
            self.transform_right = transform_right
        else:
            self.transform_right = transform_left
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset.__getitem__(idx)

        aug_1 = self.transform_left(sample)
        aug_2 = self.transform_right(sample)
        return {"aug1": aug_1, "aug2": aug_2, "target": target}
