from typing import Callable, Optional
from collections import OrderedDict

import torchvision

from catalyst.dl import ConfigExperiment


class MNIST(torchvision.datasets.MNIST):
    """MNIST Dataset with key-value ``__getitem__`` output."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        image_key: str = "image",
        target_key: str = "target",
    ):
        """
        Args:
            root (str): root directory of dataset where
                ``MNIST/processed/training.pt`` and
                ``MNIST/processed/test.pt`` exist
            train (bool): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): a function/transform that takes
                in an PIL image and returns a transformed version.
                E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it
            download (bool): If true, downloads the dataset from the internet
                and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again
            image_key (str): key to place an image
            target_key (str): key to place target
        """
        super().__init__(root, train, transform, target_transform, download)
        self.image_key = image_key
        self.target_key = target_key

    def __getitem__(self, index: int):
        """Fetch a data sample for a given index.

        Args:
            index (int): index of the element in the dataset

        Returns:
            Single element by index
        """
        image, target = self.data[index], self.targets[index]

        dict_ = {
            self.image_key: image,
            self.target_key: target,
        }

        if self.transform is not None:
            dict_ = self.transform(dict_)
        return dict_


# data loaders & transforms
class MnistGanExperiment(ConfigExperiment):
    """Simple MNIST experiment."""

    def get_datasets(
        self, stage: str, image_key: str = "image", target_key: str = "target"
    ):
        """Provides train/validation subsets from MNIST dataset.

        Args:
            stage (str): stage name e.g. ``'stage1'`` or ``'infer'``
            image_key (str):
            target_key (str):
        """
        datasets = OrderedDict()

        for dataset_name in ("train", "valid"):
            datasets[dataset_name] = MNIST(
                root="./data",
                train=(dataset_name == "train"),
                download=True,
                image_key=image_key,
                target_key=target_key,
                transform=self.get_transforms(
                    stage=stage, dataset=dataset_name
                ),
            )

        return datasets
