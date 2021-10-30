# flake8: noqa
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import pickle

import numpy as np

import torch
import torch.utils.data as data

from catalyst.contrib.datasets.functional import _check_integrity, download_and_extract_archive
from catalyst.data.dataset.metric_learning import MetricLearningTrainDataset, QueryGalleryDataset


class StandardTransform(object):
    def __init__(
        self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can " "be passed as argument"
            )

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self) -> str:
        return ""


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not _check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # @TODO: here is the channle - no image requirements!
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not _check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class Cifar10MLDataset(MetricLearningTrainDataset, CIFAR10):
    """
    Simple wrapper for CIFAR10 dataset for metric learning train stage.
    This dataset can be used only for training. For test stage
    use CIFAR10QGDataset.
    For this dataset we use only training part of the CIFAR10 and only
    those images that are labeled as 'airplane', 'automobile', 'bird', 'cat' and 'deer'
    """

    _split = 5
    classes = [
        "0 - airplane",
        "1 - automobile",
        "2 - bird",
        "3 - cat",
        "4 - deer",
    ]

    def __init__(self, **kwargs):
        """
        Raises:
            ValueError: if train argument is False (CIFAR10MLDataset
                should be used only for training)
        """
        if "train" in kwargs:
            if kwargs["train"] is False:
                raise ValueError("CIFAR10MLDataset can be used only for training stage.")
        else:
            kwargs["train"] = True
        super(Cifar10MLDataset, self).__init__(**kwargs)
        self._filter()

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets.tolist()

    def _filter(self) -> None:
        """Filter CIFAR dataset: select images of 0, 1, 2, 3, 4 classes."""
        mask = np.array(self.targets) < self._split
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask].tolist()


class CifarQGDataset(QueryGalleryDataset):
    """
    CIFAR10 for metric learning with query and gallery split.
    CIFAR10QGDataset should be used for test stage.
    For this dataset we used only test part of the CIFAR10 and only
    those images that are labeled as 'dog', 'frog', 'horse', 'ship', 'truck'.
    """

    _split = 5
    classes = [
        "5 - dog",
        "6 - frog",
        "7 - horse",
        "8 - ship",
        "9 - truck",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        gallery_fraq: Optional[float] = 0.2,
        **kwargs
    ) -> None:
        """
        Args:
            root: root directory for storing dataset
            transform: transform
            gallery_fraq: gallery size
        """
        self._cifar = CIFAR10(root, train=False, download=True, transform=transform)
        self._filter()

        self._gallery_size = int(gallery_fraq * len(self._cifar))
        self._query_size = len(self._cifar) - self._gallery_size

        self._is_query = torch.zeros(len(self._cifar)).type(torch.bool)
        self._is_query[: self._query_size] = True

    def _filter(self) -> None:
        """Filter CIFAR10 dataset: select images of 'dog', 'frog',
        'horse', 'ship', 'truck' classes."""
        mask = np.array(self._cifar.targets) >= self._split
        self._cifar.data = self._cifar.data[mask]
        self._cifar.targets = np.array(self._cifar.targets)[mask].tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item method for dataset

        Args:
            idx: index of the object

        Returns:
            Dict with features, targets and is_query flag
        """
        image, label = self._cifar[idx]
        return {
            "features": image,
            "targets": label,
            "is_query": self._is_query[idx],
        }

    def __len__(self) -> int:
        """Length"""
        return len(self._cifar)

    def __repr__(self) -> None:
        """Print info about the dataset"""
        return self._cifar.__repr__()

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
        """Images from CIFAR10"""
        return self._cifar.data

    @property
    def targets(self) -> torch.Tensor:
        """Labels of digits"""
        return self._cifar.targets


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


__all__ = ["CIFAR10", "CIFAR100"]
