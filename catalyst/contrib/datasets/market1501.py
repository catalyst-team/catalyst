from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np

import torch

from catalyst.data import MetricLearningTrainDataset, QueryGalleryDataset
from catalyst.utils import imread


class Market1501MLDataset(MetricLearningTrainDataset):
    """
    Market1501 train dataset. This dataset should be used for training
    stage of the metric learning pipeline.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Market1501 dataset for train stage of reid task.

        Args:
            root: path to a directory that contains Market-1501-v15.09.15
            transform: transformation that should be applied to images
        """
        self.root = Path(root)
        self._data_dir = self.root / "Market-1501-v15.09.15/bounding_box_train"
        self.transform = transform
        self.data, self.targets = self._load_data(self._data_dir)

    @staticmethod
    def _load_data(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load data from train directory of the dataset.
        Parse names of images to get person id as labels.

        Args:
            data_dir: path to directory that contains training data

        Returns:
            images for training and their labels
        """
        file_names = list(data_dir.glob("*.jpg"))
        data = (
            torch.from_numpy(
                np.array([imread(file_name) for file_name in file_names])
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        targets = torch.from_numpy(
            np.array(
                [int(file_name.name.split("_")[0]) for file_name in file_names]
            )
        )
        return data, targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            index: index of the element

        Returns:
            image and its label
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        """Get len of the dataset"""
        return len(self.targets)

    def get_labels(self) -> List[int]:
        """Get list of labels of dataset"""
        return self.targets.tolist()


class Market1501QGDataset(QueryGalleryDataset):
    """Market1501QGDataset is a dataset for test stage of reid pipeline"""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Market1501 dataset for testing stage of reid task.

        Args:
            root: path to a directory that contains Market-1501-v15.09.15
            transform: transformation that should be applied to images
        """
        self.root = Path(root)
        self._gallery_dir = (
            self.root / "Market-1501-v15.09.15/bounding_box_test"
        )
        self._query_dir = self.root / "Market-1501-v15.09.15/query"
        self.transform = transform
        query_data, query_targets, query_cameras = self._load_data(
            self._query_dir
        )
        gallery_data, gallery_targets, gallery_cameras = self._load_data(
            self._gallery_dir
        )

        self._query_size = query_data.shape[0]
        self._gallery_size = gallery_data.shape[0]

        self.data = torch.cat([gallery_data, query_data])
        self.pids = np.concatenate([gallery_targets, query_targets])
        self.cids = np.concatenate([gallery_cameras, query_cameras])
        self._is_query = torch.cat(
            [
                torch.zeros(size=(self._gallery_size,)),
                torch.ones(size=(self._query_size,)),
            ]
        )

    @property
    def query_size(self) -> int:
        """
        Length of query part of the dataset

        Returns:
            query size
        """
        return self._query_size

    @property
    def gallery_size(self) -> int:
        """
        Length of gallery part of the dataset

        Returns:
            gallery size
        """
        return self._gallery_size

    @staticmethod
    def _load_data(data_dir: Path) -> Tuple[torch.Tensor, Iterable, Iterable]:
        """
        Load data from directory.
        Parse names of images to get person ids as labels and camera ids.

        Args:
            data_dir: path to directory that contains data

        Returns:
            images, their labels and ids of the cameras that made the photos
        """
        file_names = list(data_dir.glob("*.jpg"))
        data = (
            torch.from_numpy(
                np.array([imread(file_name) for file_name in file_names])
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        targets = np.array(
            [int(file_name.name.split("_")[0]) for file_name in file_names]
        )
        cameras = np.array(
            [
                int(file_name.name.split("_")[1][1:2])
                for file_name in file_names
            ]
        )
        return data, targets, cameras

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item from dataset

        Args:
            index: index of the item to get

        Returns:
            dict of features, label (contains knowledge about pid and cid)
                and is_query flag that shows if the image should be used as
                query or gallery sample.
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        item = {
            "features": img,
            "pid": self.pids[index],
            "cid": self.cids[index],
            "is_query": self._is_query[index],
        }
        return item

    def __len__(self):
        """Get len of the dataset"""
        return len(self.pids)
