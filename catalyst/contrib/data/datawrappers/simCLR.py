from torch.utils.data import Dataset, IterableDataset


class simCLRDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, transforms):
        super().__init__()
        self.transforms = transforms
        self._dataset = dataset
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset.__getitem__(idx)

        batch = batch[0]
        batch1 = self.transforms(batch)
        batch2 = self.transforms(batch)
        return {"image_aug1": batch1, "image_aug2": batch2}
