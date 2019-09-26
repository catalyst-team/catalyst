from torch.utils.data import Dataset


class SomeDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        pass
