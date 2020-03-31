from torch.utils.data import Dataset


class SomeDataset(Dataset):
    """
    Class representing a `Dataset`
    """

    def __init__(self):
        """
        @TODO: Docs. Contribution is welcome
        """
        pass

    def __getitem__(self, index):
        """
        Fetch a data sample for a given index
        """
        raise NotImplementedError

    def __len__(self):
        """
        Return the size of the dataset
        """
        pass
