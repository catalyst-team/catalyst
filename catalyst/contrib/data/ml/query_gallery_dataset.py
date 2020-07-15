from typing import Dict
from abc import ABC, abstractmethod
import os
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class QueryGalleryDataset(Dataset, ABC):
    """
    QueryGallleryDataset for CMCScoreCallback
    """

    @abstractmethod
    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Dataset for query/gallery split should
        return dict with `embeddings`, `labels` and
        `is_query` key. Value by key `is_query` should
        be boolean and indicate whether current object
        is in query or in gallery.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def query_size(self) -> int:
        """
        Query/Gallery dataset should have property
        query size
        Returns:
            query size
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gallery_size(self) -> int:
        """
        Query/Gallery dataset should have property
        gallery size
        Returns:
            gallery size
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
