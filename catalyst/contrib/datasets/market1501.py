from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path, PosixPath

import gdown
import numpy as np

import torch

from catalyst.contrib.datasets.functional import extract_archive
from catalyst.data import MetricLearningTrainDataset, QueryGalleryDataset
from catalyst.data.cv.reader import imread


class Market1501MLDataset(MetricLearningTrainDataset):
    """
    Market1501 train dataset. This dataset should be used for training
    stage of the metric learning pipeline.
    """

    SOURCE_URL = "https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """

        Args:
            root:
            transform:
            target_transform:
            download:
        """
        self.root = Path(root)
        self._data_dir = self.root / "Market-1501-v15.09.15/bounding_box_train"
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.data, self.targets = self._load_data(self._data_dir)

    def download(self) -> None:
        """
        Download and unpack dataset's zip

        Raises:
            FileExistsError: if the dataset already exists in the self.root
                directory
        """
        dest_path = self.root / "Market-1501-v15.09.15.zip"
        if dest_path.is_dir():
            raise FileExistsError(f"{dest_path} already exists")
        gdown.download(url=self.SOURCE_URL, output=str(dest_path), quiet=False)
        extract_archive(
            from_path=str(dest_path),
            to_path=str(self.root),
            remove_finished=True,
        )

    @staticmethod
    def _load_data(data_dir: PosixPath) -> Tuple[torch.Tensor, Iterable]:
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
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
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        """Get len of the dataset"""
        return len(self.targets)

    def get_labels(self) -> List[int]:
        return self.targets.tolist()


class Market1501QGDataset(QueryGalleryDataset):
    SOURCE_URL = "https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """

        Args:
            root:
            transform:
            download:
        """
        self.root = Path(root)
        self._gallery_dir = (
            self.root / "Market-1501-v15.09.15/bounding_box_test"
        )
        self._query_dir = self.root / "Market-1501-v15.09.15/query"
        self.transform = transform

        if download:
            self.download()

        query_data, query_targets, query_cameras = self._load_data(
            self._query_dir
        )
        gallery_data, gallery_targets, gallery_cameras = self._load_data(
            self._gallery_dir
        )

        self._query_size = query_data.shape[0]
        self._gallery_size = gallery_data.shape[0]

        self.data = torch.cat([gallery_data, query_data])
        self.targets = np.concatenate([gallery_targets, query_targets])
        self.cameras = np.concatenate([gallery_cameras, query_cameras])
        self._is_query = torch.cat(
            [
                torch.zeros(size=(self._gallery_size,)),
                torch.ones(size=(self._query_size,)),
            ]
        )

    @property
    def query_size(self):
        return self._query_size

    @property
    def gallery_size(self):
        return self._gallery_size

    def download(self):
        dest_path = self.root / "Market-1501-v15.09.15.zip"
        if dest_path.is_dir():
            raise FileExistsError(f"{dest_path} already exists")
        gdown.download(url=self.SOURCE_URL, output=str(dest_path), quiet=False)
        extract_archive(
            from_path=str(dest_path),
            to_path=str(self.root),
            remove_finished=True,
        )

    @staticmethod
    def _load_data(
        data_dir: PosixPath,
    ) -> Tuple[torch.Tensor, Iterable, Iterable]:
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
        Get
        Args:
            index:

        Returns:

        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        item = {
            "features": img,
            "targets": {
                "pid": self.targets[index],
                "cam_id": self.cameras[index],
            },
            "is_query": self._is_query[index],
        }
        return item

    def __len__(self):
        """Get len of the dataset"""
        return len(self.targets)
