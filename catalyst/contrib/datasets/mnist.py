# flake8: noqa
from typing import Any, Callable, Dict, List, Optional
import os

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets.functional import (
    download_and_extract_archive,
    read_image_file,
    read_label_file,
)
from catalyst.data.dataset.metric_learning import (
    MetricLearningTrainDataset,
    QueryGalleryDataset,
)


class MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset."""

    _repr_indent = 4

    resources = [
        (
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Args:
            root (string): Root directory of dataset where
                ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from
                ``training.pt``, otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from
                the internet and puts it in root directory. If dataset
                is already downloaded, it is not downloaded again.
            transform (callable, optional): A function/transform that
                takes in an image and returns a transformed version.
            target_transform (callable, optional): A function/transform
                that takes in the target and transforms it.
        """
        if isinstance(root, torch._six.string_classes):  # noqa: WPS437
            root = os.path.expanduser(root)
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index].numpy(), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """@TODO: Docs. Contribution is welcome."""
        return len(self.data)

    def __repr__(self):
        """@TODO: Docs. Contribution is welcome."""
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @property
    def raw_folder(self):
        """@TODO: Docs. Contribution is welcome."""
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        """@TODO: Docs. Contribution is welcome."""
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self):
        """@TODO: Docs. Contribution is welcome."""
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # process and save as torch files
        print("Processing...")

        training_set = (
            read_image_file(
                os.path.join(self.raw_folder, "train-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.raw_folder, "train-labels-idx1-ubyte")
            ),
        )
        test_set = (
            read_image_file(
                os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")
            ),
        )
        with open(
            os.path.join(self.processed_folder, self.training_file), "wb"
        ) as f:
            torch.save(training_set, f)
        with open(
            os.path.join(self.processed_folder, self.test_file), "wb"
        ) as f:
            torch.save(test_set, f)

        print("Done!")

    def extra_repr(self):
        """@TODO: Docs. Contribution is welcome."""
        return "Split: {}".format("Train" if self.train is True else "Test")


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
            root: root directory for storing dataset
            transform: transform
            gallery_fraq: gallery size
        """
        self._mnist = MNIST(
            root, train=False, download=True, transform=transform
        )

        self._gallery_size = int(gallery_fraq * len(self._mnist))
        self._query_size = len(self._mnist) - self._gallery_size

        self._is_query = torch.zeros(len(self._mnist)).type(torch.bool)
        self._is_query[: self._query_size] = True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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

    def __len__(self) -> int:
        """Length"""
        return len(self._mnist)

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size


__all__ = ["MNIST", "MnistMLDataset", "MnistQGDataset"]
