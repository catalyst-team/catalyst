import numpy as np
from torch.utils.data import Dataset, IterableDataset


class simCLRDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, transforms):
        super().__init__()
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset.__getitem__(idx)

        aug_1 = self.transforms(sample)
        aug_2 = self.transforms(sample)
        return {"aug1": aug_1, "aug2": aug_2, "target": target}
