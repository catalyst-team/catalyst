from abc import ABC, abstractmethod
from typing import List, Optional, Callable

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets import MNIST
from catalyst.data.dataset import QueryGalleryDataset


class MetricLearningTrainDataset(Dataset, ABC):
    """
    Base class for datasets adapted for
    metric learning train stage.
    """

    @abstractmethod
    def get_labels(self) -> List[int]:
        """
        Dataset for metric learning must provide
        label of each sample for forming positive
        and negative pairs during the training
        based on these labels.

        Returns:
            labels of samples
        """
        raise NotImplementedError


class MnistMLDataset(MetricLearningTrainDataset, MNIST):
    """
    Simple wrapper for MNIST dataset
    """

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets.tolist()


class MnistQGDataset(QueryGalleryDataset):
    """MNIST for metric learning with query and gallery split"""

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            gallery_fraq: Optional[float] = 0.2,
    ) -> None:
        """

        Args:
            gallery_fraq: gallery size
            **mnist_args: args for MNIST dataset
                (see catalyst.contrib.datasets.MNIST)
        """
        self._mnist = MNIST(
            root, train=False, download=True, transform=transform
        )

        self._gallery_size = int(gallery_fraq * len(self._mnist))
        self._query_size = len(self._mnist) - self._gallery_size

        self._is_query = torch.zeros(len(self._mnist)).type(torch.bool)
        self._is_query[: self._query_size] = True

    def __getitem__(self, idx):
        """
        Get item method for dataset


        Args:
            idx: index of the object

        Returns:
            Dict with features, targets and is_query flag
        """
        image, label = self._mnist[idx]
        return {
            "features": image,
            "targets": label,
            "is_query": self._is_query[idx],
        }

    def __len__(self):
        """Length"""
        return len(self._mnist)

    @property
    def gallery_size(self):
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self):
        """Query Gallery dataset should have query_size property"""
        return self._query_size
