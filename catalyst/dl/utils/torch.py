from typing import Callable, Iterable

import torch
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.data import ListDataset


def get_loader(
    data_source: Iterable[dict],
    open_fn: Callable,
    dict_transform: Callable = None,
    sampler=None,
    collate_fn: Callable = default_collate_fn,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False,
):
    """Creates a DataLoader from given source and its open/transform params.

    Args:
        data_source (Iterable[dict]): and iterable containing your
            data annotations,
            (for example path to images, labels, bboxes, etc)
        open_fn (Callable): function, that can open your
            annotations dict and
            transfer it to data, needed by your network
            (for example open image by path, or tokenize read string)
        dict_transform (callable): transforms to use on dict
            (for example normalize image, add blur, crop/resize/etc)
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset
        batch_size (int, optional): how many samples per batch to load
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded
            in the main process
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool, optional): set to ``True`` to drop
            the last incomplete batch, if the dataset size is not divisible
            by the batch size. If ``False`` and the size of dataset
            is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)

    Returns:
        DataLoader with ``catalyst.data.ListDataset``
    """
    dataset = ListDataset(
        list_data=data_source, open_fn=open_fn, dict_transform=dict_transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    return loader


__all__ = ["get_loader"]
