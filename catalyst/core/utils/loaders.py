from collections import OrderedDict
from copy import copy

from torch.utils.data import DataLoader, DistributedSampler

from catalyst import utils
from catalyst.data import DistributedSamplerWrapper


def _process_dataloader(loader):
    sampler = (
        DistributedSampler(dataset=loader.dataset)
        if loader.sampler is not None
        else DistributedSamplerWrapper(sampler=loader.sampler)
    )
    loader = DataLoader(
        dataset=copy(loader.dataset),
        batch_size=loader.batch_size,
        # shuffle=loader.shuffle,
        sampler=sampler,
        # batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        # collate_fn=loader.collate_fn,
        # pin_memory=loader.pin_memory,
        # drop_last=loader.drop_last,
    )
    return loader


def process_loaders(loaders):
    rank = utils.get_rank()
    if rank >= 0:
        loaders = OrderedDict(
            [
                (key, _process_dataloader(value))
                for key, value in loaders.items()
            ]
        )
    return loaders
