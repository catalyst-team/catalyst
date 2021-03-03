from typing import Any, Callable, Dict, List, Optional
import os

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets.functional import (
    download_and_extract_archive,
    read_sn3_pascalvincent_tensor,
)
from catalyst.data.dataset.metric_learning import MetricLearningTrainDataset, QueryGalleryDataset


def _read_label_file(path):
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def _read_image_file(path):
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x


class MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset."""

    _repr_indent = 4

    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    resources = [
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
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
        self, root, train=True, transform=None, target_transform=None, download=False,
    ):
        """
        Args:
            root: Root directory of dataset where
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
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index: Index

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
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

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
            _read_image_file(os.path.join(self.raw_folder, "train-images-idx3-ubyte")),
            _read_label_file(os.path.join(self.raw_folder, "train-labels-idx1-ubyte")),
        )
        test_set = (
            _read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
            _read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")),
        )
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")

    def extra_repr(self):
        """@TODO: Docs. Contribution is welcome."""
        return "Split: {}".format("Train" if self.train is True else "Test")


class MnistMLDataset(MetricLearningTrainDataset, MNIST):
    """
    Simple wrapper for MNIST dataset for metric learning train stage.
    This dataset can be used only for training. For test stage
    use MnistQGDataset.

    For this dataset we use only training part of the MNIST and only
    those images that are labeled as 0, 1, 2, 3, 4.
    """

    _split = 5
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
    ]

    def __init__(self, **kwargs):
        """
        Raises:
            ValueError: if train argument is False (MnistMLDataset
                should be used only for training)
        """
        if "train" in kwargs:
            if kwargs["train"] is False:
                raise ValueError("MnistMLDataset can be used only for training stage.")
        else:
            kwargs["train"] = True
        super(MnistMLDataset, self).__init__(**kwargs)
        self._filter()

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets.tolist()

    def _filter(self) -> None:
        """Filter MNIST dataset: select images of 0, 1, 2, 3, 4 classes."""
        mask = self.targets < self._split
        self.data = self.data[mask]
        self.targets = self.targets[mask]


class MnistQGDataset(QueryGalleryDataset):
    """
    MNIST for metric learning with query and gallery split.
    MnistQGDataset should be used for test stage.

    For this dataset we used only test part of the MNIST and only
    those images that are labeled as 5, 6, 7, 8, 9.
    """

    _split = 5
    classes = [
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self, root: str, transform: Optional[Callable] = None, gallery_fraq: Optional[float] = 0.2,
    ) -> None:
        """
        Args:
            root: root directory for storing dataset
            transform: transform
            gallery_fraq: gallery size
        """
        self._mnist = MNIST(root, train=False, download=True, transform=transform)
        self._filter()

        self._gallery_size = int(gallery_fraq * len(self._mnist))
        self._query_size = len(self._mnist) - self._gallery_size

        self._is_query = torch.zeros(len(self._mnist)).type(torch.bool)
        self._is_query[: self._query_size] = True

    def _filter(self) -> None:
        """Filter MNIST dataset: select images of 5, 6, 7, 8, 9 classes."""
        mask = self._mnist.targets >= self._split
        self._mnist.data = self._mnist.data[mask]
        self._mnist.targets = self._mnist.targets[mask]

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

    def __repr__(self) -> None:
        """Print info about the dataset"""
        return self._mnist.__repr__()

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size

    @property
    def data(self) -> torch.Tensor:
        """Images from MNIST"""
        return self._mnist.data

    @property
    def targets(self) -> torch.Tensor:
        """Labels of digits"""
        return self._mnist.targets


__all__ = ["MNIST", "MnistMLDataset", "MnistQGDataset"]
