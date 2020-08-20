from typing import Dict, List
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


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

        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        """
        raise NotImplementedError()


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
        raise NotImplementedError()

    @property
    @abstractmethod
    def query_size(self) -> int:
        """
        Query/Gallery dataset should have property
        query size.

        Returns:
            query size  # noqa: DAR202

        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def gallery_size(self) -> int:
        """
        Query/Gallery dataset should have property
        gallery size.

        Returns:
            gallery size  # noqa: DAR202

        Raises:
            NotImplementedError: You should implement it  # noqa: DAR402
        """
        raise NotImplementedError()


__all__ = ["MetricLearningTrainDataset", "QueryGalleryDataset"]
