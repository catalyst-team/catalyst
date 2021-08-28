import numpy as np
from torch.utils.data import Dataset, IterableDataset


class simCLRDatasetWrapper(Dataset):
    def __init__(
        self, dataset: Dataset, transforms=None, transform_left=None, transform_right=None
    ):
        super().__init__()

        if not transform_right is None and not transform_left is None:
            self.transform_right = transform_right
            self.transform_left = transform_left
        elif not transforms is None:
            self.transform_right = transforms
            self.transform_left = transforms
        else:
            raise ValueError(
                "Specify transforms or transform_left and transform_right simultaneously."
            )
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset.__getitem__(idx)

        aug_1 = self.transform_left(sample)
        aug_2 = self.transform_right(sample)
        return {"aug1": aug_1, "aug2": aug_2, "target": target}
