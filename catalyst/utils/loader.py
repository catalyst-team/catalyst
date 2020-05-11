from typing import Dict, Union  # isort:skip

from torch.utils.data import DataLoader


def get_native_batch_from_loader(loader: DataLoader, batch_index: int = 0):
    """
    Returns a batch from experiment loader

    Args:
        loader (DataLoader): Loader to get batch from
        batch_index (int): Index of batch to take from dataset of the loader
    """
    dataset = loader.dataset
    collate_fn = loader.collate_fn
    return collate_fn([dataset[batch_index]])


def get_native_batch_from_loaders(
    loaders: Dict[str, DataLoader],
    loader: Union[str, int] = 0,
    batch_index: int = 0,
):
    """
    Returns a batch from experiment loaders by its index or name.

    Args:
        loaders (Dict[str, DataLoader]): Loaders list to get loader from
        loader (Union[str, int]): Loader name or its index, default is zero
        batch_index (int): Index of batch to take from dataset of the loader
    """
    if isinstance(loader, str):
        _loader = loaders[loader]
    elif isinstance(loader, int):
        _loader = list(loaders.values())[loader]
    else:
        raise TypeError("Loader parameter must be a string or an integer")

    return get_native_batch_from_loader(
        loader=_loader, batch_index=batch_index
    )


__all__ = [
    "get_native_batch_from_loader",
    "get_native_batch_from_loaders",
]
