from typing import Any, Dict, List, Optional, Sequence
import os

import torch
from torch.utils.data import Dataset

from catalyst.contrib.data.dataset_ml import (
    MetricLearningTrainDataset,
    QueryGalleryDataset,
)
from catalyst.contrib.datasets.misc import (
    download_and_extract_archive,
    read_sn3_pascalvincent_tensor,
)


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
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset for testing purposes.

     Args:
        root: Root directory of dataset where
            ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from
            ``training.pt``, otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from
            the internet and puts it in root directory. If dataset
            is already downloaded, it is not downloaded again.
        normalize (tuple, optional): mean and std
            for the MNIST dataset normalization.
        numpy (bool, optional): boolean flag to return an np.ndarray,
            rather than torch.tensor (default: False).

    Raises:
        RuntimeError: If ``download is False`` and the dataset not found.
    """

    _repr_indent = 4

    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    resources = [
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",  # noqa: E501, W505
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",  # noqa: E501, W505
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",  # noqa: E501, W505
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",  # noqa: E501, W505
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    cache_folder = "MNIST"
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
        root: str,
        train: bool = True,
        download: bool = True,
        normalize: tuple = (0.1307, 0.3081),
        numpy: bool = False,
    ):
        """Init."""
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize
        if self.normalize is not None:
            assert len(self.normalize) == 2, "normalize should be (mean, variance)"
        self.numpy = numpy

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
        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index: Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index].float().unsqueeze(0), int(self.targets[index])
        if self.normalize is not None:
            img = self.normalize_tensor(img, *self.normalize)
        if self.numpy:
            img = img.cpu().numpy()[0]

        return img, target

    def __len__(self):
        """Length."""
        return len(self.data)

    def __repr__(self):
        """Repr."""
        head = "Dataset " + self.cache_folder
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @staticmethod
    def normalize_tensor(
        tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0
    ) -> torch.Tensor:
        """Internal tensor normalization."""
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)

    @property
    def raw_folder(self):
        """@TODO: Docs. Contribution is welcome."""
        return os.path.join(self.root, self.cache_folder, "raw")

    @property
    def processed_folder(self):
        """@TODO: Docs. Contribution is welcome."""
        return os.path.join(self.root, self.cache_folder, "processed")

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
        Args:
            **kwargs: Keyword-arguments passed to ``super().__init__`` method.

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

    Args:
        gallery_fraq: gallery size
        **kwargs: MNIST args
    """

    _split = 5
    classes = [
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, gallery_fraq: Optional[float] = 0.2, **kwargs) -> None:
        """Init."""
        self._mnist = MNIST(train=False, **kwargs)
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


class PartialMNIST(MNIST):
    """Partial MNIST dataset.

    Args:
        num_samples: number of examples per selected class/digit. default: 100
        classes: list selected MNIST classes. default: (0, 1, 2)
        **kwargs: MNIST parameters

    Examples:
        >>> dataset = PartialMNIST(".", download=True)
        >>> len(dataset)
        300
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([100, 100, 100])
    """

    def __init__(
        self,
        num_samples: int = 100,
        classes: Optional[Sequence] = (0, 1, 2),
        **kwargs,
    ):
        self.num_samples = num_samples
        self.classes = sorted(classes) if classes else list(range(10))
        super().__init__(**kwargs)
        self.data, self.targets = self._prepare_subset(
            self.data, self.targets, num_samples=self.num_samples, classes=self.classes
        )

    @staticmethod
    def _prepare_subset(
        full_data: torch.Tensor,
        full_targets: torch.Tensor,
        num_samples: int,
        classes: Sequence,
    ):
        counts = {d: 0 for d in classes}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if counts.get(label, float("inf")) >= num_samples:
                continue
            indexes.append(idx)
            counts[label] += 1
            if all(counts[k] >= num_samples for k in counts):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        return data, targets


__all__ = ["MNIST", "MnistMLDataset", "MnistQGDataset", "PartialMNIST"]
