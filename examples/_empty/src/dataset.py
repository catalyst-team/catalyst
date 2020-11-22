from torch.utils.data import Dataset


class SomeDataset(Dataset):
    """Class representing a `Dataset`."""

    def __init__(self):
        """Docs? Contribution is welcome.."""
        pass

    def __getitem__(self, index: int):
        """Fetch a data sample for a given index.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        pass
