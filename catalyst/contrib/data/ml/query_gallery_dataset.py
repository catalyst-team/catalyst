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


class QueryGalleryFolderDataset(QueryGalleryDataset):
    """
    QueryGalleryFolderDataset

    Torchvision style dataset for query gallery data.
    Expecting input: ::

        path/query/class_x/xxx.ext
        path/query/class_x/xxy.ext
        path/query/class_x/xxz.ext

        path/gallery/class_y/123.ext
        path/gallery/class_y/nsdf3.ext
        path/gallery/class_y/asd932_.ext
    """

    def __init__(self, path: str, transform=None):
        """

        Args:
            path: path to dataset
            transform: transforms for images
        """
        path = Path(path)
        query_path = path / "query"
        gallery_path = path / "gallery"
        self.query_labels = []
        self.query_imgs = []
        for label in os.listdir(query_path):
            img_path = query_path / str(label)
            for img in os.listdir(img_path):
                pil_image = Image.open(img_path / img)
                self.query_imgs.append(pil_image)
                self.query_labels.append(int(label))
        self.gallery_labels = []
        self.gallery_imgs = []
        for label in os.listdir(gallery_path):
            img_path = query_path / str(label)
            for img in os.listdir(img_path):
                pil_image = Image.open(img_path / img)
                self.gallery_imgs.append(pil_image)
                self.gallery_labels.append(int(label))
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        """sum of query and gallery sizes"""
        return len(self.query_labels) + len(self.gallery_labels)

    @property
    def query_size(self):
        """
        Query/Gallery dataset should have property
        gallery size
        Returns:
            gallery size
        """
        return len(self.query_labels)

    @property
    def gallery_size(self):
        """
        Query/Gallery dataset should have property
        gallery size
        Returns:
            gallery size
        """
        return len(self.gallery_labels)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """
        Getitem method
        Args:
            index:

        Returns:
            Dict of tensors
        """
        if index >= self.query_size:
            index -= self.query_size
            return {
                "features": self.transform(self.gallery_imgs[index]),
                "targets": self.gallery_labels[index],
                "is_query": False,
            }
        return {
            "features": self.transform(self.query_imgs[index]),
            "targets": self.query_labels[index],
            "is_query": True,
        }
