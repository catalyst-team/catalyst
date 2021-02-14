# flake8: noqa
from collections import OrderedDict

import torch
import torchvision
from torchvision import transforms

from catalyst import utils
from catalyst.dl import SupervisedConfigRunner


class CIFAR10(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."""

    def __getitem__(self, index: int):
        """Fetch a data sample for a given index.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        image, target = self.data[index], self.targets[index]
        if self.transform is not None:
            # image = self.transform({"image": image})["image"]
            image = self.transform(image)

        return image, target


class CustomSupervisedConfigRunner(SupervisedConfigRunner):
    """``ConfigExperiment`` with CIFAR10 dataset."""

    def get_model(self, stage: str, epoch: int = None):
        """
        Model specification for current stage

        Args:
            stage: current stage name

        Returns:
            model
        """
        model = super().get_model(stage=stage, epoch=epoch)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        conv_layres = ["conv1", "pool", "conv2"]

        if stage == "tune":
            for key in conv_layres:
                layer = getattr(model, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model

    def get_transform(self, stage: str = None, dataset: str = None):
        """Docs? Contribution is welcome"""
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_datasets(self, stage: str, epoch: int = None) -> "OrderedDict[str, Dataset]":
        """Provides train/validation subsets from CIFAR10 dataset.

        Args:
            stage: stage name e.g. ``'stage1'`` or ``'infer'``
        """
        datasets = OrderedDict()
        for dataset in ("train", "valid"):
            datasets[dataset] = CIFAR10(
                root="./data",
                train=(dataset == "train"),
                download=True,
                transform=self.get_transform(stage=stage, dataset=dataset),
            )

        return datasets
