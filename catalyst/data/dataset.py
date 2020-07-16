from typing import Any, Callable, Dict, List, Union
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, Sampler

from catalyst.utils import merge_dicts

_Path = Union[str, Path]


class ListDataset(Dataset):
    """General purpose dataset class with several data sources `list_data`."""

    def __init__(
        self,
        list_data: List[Dict],
        open_fn: Callable,
        dict_transform: Callable = None,
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
        """
        self.data = list_data
        self.open_fn = open_fn
        self.dict_transform = (
            dict_transform if dict_transform is not None else lambda x: x
        )

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.

        Args:
            index (int): index of the element in the dataset

        Returns:
            Single element by index
        """
        item = self.data[index]
        dict_ = self.open_fn(item)
        dict_ = self.dict_transform(dict_)

        return dict_

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.data)


class MergeDataset(Dataset):
    """Abstraction to merge several datasets into one dataset."""

    def __init__(self, *datasets: Dataset, dict_transform: Callable = None):
        """
        Args:
            datasets (List[Dataset]): params count of datasets to merge
            dict_transform (callable): transforms common for all datasets.
                (for example normalize image, add blur, crop/resize/etc)
        """
        self.length = len(datasets[0])
        assert all(len(x) == self.length for x in datasets)
        self.datasets = datasets
        self.dict_transform = dict_transform

    def __getitem__(self, index: int) -> Any:
        """Get item from all datasets.

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
        """
        Returns:
            int: length of the dataset
        """
        return self.length


class NumpyDataset(Dataset):
    """General purpose dataset class to use with `numpy_data`."""

    def __init__(
        self,
        numpy_data: np.ndarray,
        numpy_key: str = "features",
        dict_transform: Callable = None,
    ):
        """
        General purpose dataset class to use with `numpy_data`.

        Args:
            numpy_data (np.ndarray): numpy data
              (for example path to embeddings, features, etc.)
            numpy_key (str): key to use for output dictionary
            dict_transform (callable): transforms to use on dict.
              (for example normalize vector, etc)
        """
        super().__init__()
        self.data = numpy_data
        self.key = numpy_key
        self.dict_transform = (
            dict_transform if dict_transform is not None else lambda x: x
        )

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.

        Args:
            index (int): index of the element in the dataset

        Returns:
            Single element by index
        """
        dict_ = {self.key: np.copy(self.data[index])}
        dict_ = self.dict_transform(dict_)
        return dict_

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.data)


class PathsDataset(ListDataset):
    """
    Dataset that derives features and targets from samples filesystem paths.

    Examples:
        >>> label_fn = lambda x: x.split("_")[0]
        >>> dataset = PathsDataset(
        >>>     filenames=Path("/path/to/images/").glob("*.jpg"),
        >>>     label_fn=label_fn,
        >>>     open_fn=open_fn,
        >>> )
    """

    def __init__(
        self,
        filenames: List[_Path],
        open_fn: Callable[[dict], dict],
        label_fn: Callable[[_Path], Any],
        **list_dataset_params
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
        """
        list_data = [
            {"features": filename, "targets": label_fn(filename)}
            for filename in filenames
        ]

        super().__init__(
            list_data=list_data, open_fn=open_fn, **list_dataset_params
        )


class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler: Sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index (int): index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class QueryGalleryDataset(Dataset, ABC):
    """
    QueryGallleryDataset for CMCScoreCallback
    """

    @abstractmethod
    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Dataset for query/gallery split should
        return dict with `feature`, `targets` and
        `is_query` key. Value by key `is_query` should
        be boolean and indicate whether current object
        is in query or in gallery.
        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def query_size(self) -> int:
        """
        Query/Gallery dataset should have property
        query size
        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        Returns:
            query size
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gallery_size(self) -> int:
        """
        Query/Gallery dataset should have property
        gallery size
        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        Returns:
            gallery size
        """
        raise NotImplementedError


__all__ = [
    "ListDataset",
    "MergeDataset",
    "NumpyDataset",
    "PathsDataset",
    "DatasetFromSampler",
]
