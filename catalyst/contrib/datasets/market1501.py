from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

from catalyst.contrib.utils import imread
from catalyst.data import MetricLearningTrainDataset, QueryGalleryDataset


class Market1501MLDataset(MetricLearningTrainDataset):
    """
    Market1501 train dataset. This dataset should be used for training stage of the reid pipeline.

    .. _Scalable Person Re-identification\: A Benchmark:
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf
    """

    def __init__(
        self, root: str, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        self.images, self.pids = self._load_data(self._data_dir)

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
        filenames = list(data_dir.glob("*.jpg"))
        data = torch.from_numpy(np.array([imread(filename) for filename in filenames])).float()
        targets = torch.from_numpy(
            np.array([int(filename.name.split("_")[0]) for filename in filenames])
        )
        return data, targets

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            index: index of the element
        Returns:
            dict of image and its pid
        """
        image, pid = self.images[index], self.pids[index]
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "pid": pid}

    def __len__(self) -> int:
        """Get len of the dataset"""
        return len(self.pids)

    def get_labels(self) -> List[int]:
        """Get list of labels of dataset"""
        return self.pids.tolist()


class Market1501QGDataset(QueryGalleryDataset):
    """Market1501QGDataset is a dataset for test stage of reid pipeline"""

    def __init__(
        self, root: str, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Market1501 dataset for testing stage of reid task.

        Args:
            root: path to a directory that contains Market-1501-v15.09.15
            transform: transformation that should be applied to images
        """
        self.root = Path(root)
        self._gallery_dir = self.root / "Market-1501-v15.09.15/bounding_box_test"
        self._query_dir = self.root / "Market-1501-v15.09.15/query"
        self.transform = transform
        query_data, query_pids, query_cids = self._load_data(self._query_dir)
        gallery_data, gallery_pids, gallery_cids = self._load_data(self._gallery_dir)

        self._query_size = query_data.shape[0]
        self._gallery_size = gallery_data.shape[0]

        self.data = torch.cat([gallery_data, query_data])
        self.pids = np.concatenate([gallery_pids, query_pids])
        self.cids = np.concatenate([gallery_cids, query_cids])
        self._is_query = torch.cat(
            [torch.zeros(size=(self._gallery_size,)), torch.ones(size=(self._query_size,))]
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
        # Gallery dataset contains good, junk and distractor images;
        # junk ones (marked as -1) should be neglected during evaluation.
        filenames = list(data_dir.glob("[!-]*.jpg"))
        data = torch.from_numpy(np.array([imread(filename) for filename in filenames])).float()
        pids = np.array([int(filename.name.split("_")[0]) for filename in filenames])
        cids = np.array([int(filename.name.split("_")[1][1:2]) for filename in filenames])
        return data, pids, cids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item from dataset

        Args:
            index: index of the item to get
        Returns:
            dict of image, pid, cid and is_query flag that shows if the image should be used as
            query or gallery sample.
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        item = {
            "image": img,
            "pid": self.pids[index],
            "cid": self.cids[index],
            "is_query": self._is_query[index],
        }
        return item

    def __len__(self):
        """Get len of the dataset"""
        return len(self.pids)


__all__ = ["Market1501MLDataset", "Market1501QGDataset"]
