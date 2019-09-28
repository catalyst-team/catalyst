import random
from pathlib import Path
from typing import List, Dict, Callable, Any, Union

from catalyst.utils.misc import merge_dicts
from torch.utils.data import Dataset

_Path = Union[str, Path]


class ListDataset(Dataset):
    """
    General purpose dataset class with several data sources `list_data`
    """
    def __init__(
        self,
        list_data: List[Dict],
        open_fn: Callable,
        dict_transform: Callable = None,
        cache_prob: float = -1,
        cache_transforms: bool = False
    ):
        """
        Args:
            list_data (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            open_fn (callable): function, that can open your
                annotations dict and
                transfer it to data, needed by your network
                (for example open image by path, or tokenize read string.)
            dict_transform (callable): transforms to use on dict.
                (for example normalize image, add blur, crop/resize/etc)
            cache_prob (float): probability of saving opened dict to RAM
                for speedup
            cache_transforms (bool): flag if you need
                to cache sample after transformations to RAM
        """
        self.data = list_data
        self.open_fn = open_fn
        self.dict_transform = dict_transform
        self.cache_prob = cache_prob
        self.cache_transforms = cache_transforms
        self.cache = dict()

    def prepare_new_item(self, index: int):
        row = self.data[index]
        dict_ = self.open_fn(row)

        if self.cache_transforms and self.dict_transform is not None:
            dict_ = self.dict_transform(dict_)

        return dict_

    def prepare_item_from_cache(self, index: int):
        return self.cache.get(index, None)

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
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

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.data)


class MergeDataset(Dataset):
    """
    Abstraction to merge several datasets into one dataset.
    """
    def __init__(self, *datasets: Dataset, dict_transform: Callable = None):
        """
        Args:
            datasets (List[Dataset]): params count of datasets to merge
            dict_transform (callable): transforms common for all datasets.
                (for example normalize image, add blur, crop/resize/etc)
        """
        self.len = len(datasets[0])
        assert all([len(x) == self.len for x in datasets])
        self.datasets = datasets
        self.dict_transform = dict_transform

    def __getitem__(self, index: int) -> Any:
        """Get item from all datasets

        Args:
            index (int): index to value from all datasets

        Returns:
            list: list of value in every dataset
        """
        dcts = [x[index] for x in self.datasets]
        dct = merge_dicts(*dcts)

        if self.dict_transform is not None:
            dct = self.dict_transform(dct)

        return dct

    def __len__(self) -> int:
        return self.len


class PathsDataset(ListDataset):
    """
    Dataset that derives features and targets from samples filesystem paths.
    """
    def __init__(
        self, filenames: List[_Path], open_fn: Callable[[dict], dict],
        label_fn: Callable[[_Path], Any], **list_dataset_params
    ):
        """
         Args:
            filenames (List[str]): list of file paths that store information
                about your dataset samples; it could be images, texts or
                any other files in general.
            open_fn (callable): function, that can open your
                annotations dict and
                transfer it to data, needed by your network
                (for example open image by path, or tokenize read string)
            label_fn (callable): function, that can extract target
                value from sample path
                (for example, your sample could be an image file like
                ``/path/to/your/image_1.png`` where the target is encoded as
                a part of file path)
            list_dataset_params (dict): base class initialization
                parameters.

        Examples:
            >>> label_fn = lambda x: x.split("_")[0]
            >>> dataset = PathsDataset(
            >>>     filenames=Path("/path/to/images/").glob("*.jpg"),
            >>>     label_fn=label_fn,
            >>>     open_fn=open_fn,
            >>> )
        """
        list_data = [
            dict(features=filename, targets=label_fn(filename))
            for filename in filenames
        ]

        super().__init__(
            list_data=list_data, open_fn=open_fn, **list_dataset_params
        )


__all__ = ["_Path", "ListDataset", "MergeDataset", "PathsDataset"]
