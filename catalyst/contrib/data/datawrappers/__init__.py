from torch.utils.data import Dataset, IterableDataset


class simCLRDatasetWrapper(IterableDataset):
    def __init__(self, loader: Dataset, transforms):
        super().__init__()
        self.transforms = transforms
        self.dataset = loader

    def __iter__(self):
        """Iterator.
        Returns:
            iterator object
        """
        self._iterator = iter(self.dataset)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __next__(self):
        """Next batch.
        Returns:
            next batch
        """
        batch = next(self._iterator)[0]
        batch1 = self.transforms(batch)
        batch2 = self.transforms(batch)
        return {"image_aug1": batch1, "image_aug2": batch2}
