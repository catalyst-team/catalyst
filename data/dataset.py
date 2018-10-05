import random
from typing import List, Dict, Callable
from torch.utils.data import Dataset
from catalyst.utils.misc import merge_dicts


class ListDataset(Dataset):
    """
    General purpose dataset class.
    """
    def __init__(
            self,
            list_data: List[Dict],
            open_fn: Callable,
            dict_transform: Callable = None,
            cache_prob: float = -1,
            cache_transforms: bool = False):
        """

        :param list_data: list of dicts, that stores you data annotations,
            for example path to images, labels, bboxes, etc.
        :param open_fn: function, that can open your annotations dict and
            transfer it to data, needed by your network
            for example open image by path, or read string and tokenize it.
        :param dict_transform: transforms to use on dict.
            for example normalize image by ImageNet, add blur, crop/resize/etc
        :param cache_prob: probability of saving opened dict to RAM for speedup
        :param cache_transforms: boolean flag if you need
            to cache sample after transformations to RAM
        """
        self.data = list_data
        self.open_fn = open_fn
        self.dict_transform = dict_transform
        self.cache_prob = cache_prob
        self.cache_transforms = cache_transforms
        self.cache = dict()

    def prepare_new_item(self, index):
        row = self.data[index]
        dict_ = self.open_fn(row)

        if self.cache_transforms and self.dict_transform is not None:
            dict_ = self.dict_transform(dict_)

        return dict_

    def prepare_item_from_cache(self, index):
        return self.cache.get(index, None)

    def __getitem__(self, index):
        dict_ = None

        if random.random() < self.cache_prob:
            dict_ = self.prepare_item_from_cache(index)

        if dict_ is None:
            dict_ = self.prepare_new_item(index)
            if self.cache_prob > 0:
                self.cache[index] = dict_

        if not self.cache_transforms and self.dict_transform is not None:
            dict_ = self.dict_transform(dict_)

        return dict_

    def __len__(self):
        return len(self.data)


class MergeDataset(Dataset):
    def __init__(self, *datasets):
        self.len = len(datasets[0])
        assert all([len(x) == self.len for x in datasets])
        self.datasets = datasets

    def __getitem__(self, index):
        dcts = [x[index] for x in self.datasets]
        dct = merge_dicts(*dcts)
        return dct

    def __len__(self):
        return self.len
