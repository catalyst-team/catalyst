from typing import Dict, Union  # isort:skip

from torch.utils.data import DataLoader


def get_native_batch(
        loaders: Dict[str, DataLoader],
        loader: Union[str, int] = 0,
        data_index: int = 0
):
    """
    Returns a batch from experiment loader

    Args:
        loaders (Dict[str, DataLoader]): Loaders list to get loader from
        loader (Union[str, int]): Loader name or its index, default is zero
        data_index (int): Index of batch to take from dataset of the loader
    """
    if isinstance(loader, str):
        _loader = loaders[loader]
    elif isinstance(loader, int):
        _loader = list(loaders.values())[loader]
    else:
        raise TypeError("Loader parameter must be a string or an integer")

    dataset = _loader.dataset
    collate_fn = _loader.collate_fn

    sample = collate_fn([dataset[data_index]])

    return sample


__all__ = ["get_native_batch"]
