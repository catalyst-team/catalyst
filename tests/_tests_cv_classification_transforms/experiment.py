from typing import Tuple
from collections import OrderedDict

from torch.utils.data import Dataset

from catalyst.contrib.datasets import MNIST as _MNIST
from catalyst.dl.experiment import ConfigExperiment


class MNIST(_MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset."""

    def __getitem__(self, index: int) -> Tuple:
        """Fetches a sample for a given index from MNIST dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform({"image": image})["image"]

        return image, target


class Experiment(ConfigExperiment):
    """``ConfigExperiment`` with MNIST dataset."""

    def get_datasets(
        self, stage: str, **kwargs
    ) -> "OrderedDict[str, Dataset]":
        """Provides train/validation subsets from MNIST dataset.

        Args:
            stage: stage name e.g. ``'stage1'`` or ``'infer'``
            **kwargs: extra params

        Returns:
            ordered dict with datasets
        """
        datasets = OrderedDict()
        for mode in ("train", "valid"):
            datasets[mode] = MNIST(
                "./data",
                train=False,
                download=True,
                transform=self.get_transforms(stage=stage, dataset=mode),
            )

        return datasets
