# flake8: noqa
from collections import OrderedDict

import torch
import torchvision
from torchvision import transforms

from catalyst import utils
from catalyst.dl import SupervisedConfigRunner
from catalyst.settings import IS_HYDRA_AVAILABLE


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


class RunnerMixin:
    def get_model(self, stage: str, epoch: int = None):
        if self.model is None:
            # first stage
            model = super().get_model(stage=stage, epoch=epoch)
        else:
            model = self.model
        conv_layres = ["conv1", "pool", "conv2"]
        if stage == "tune":
            # second stage logic
            model = self.model
            for key in conv_layres:
                layer = getattr(model, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model

    def get_transform(self, stage: str = None, dataset: str = None):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_datasets(self, stage: str, epoch: int = None) -> "OrderedDict[str, Dataset]":
        """Provides train/validation datasets from CIFAR10 dataset."""
        datasets = OrderedDict()
        for dataset in ("train", "valid"):
            datasets[dataset] = CIFAR10(
                root="./data",
                train=(dataset == "train"),
                download=True,
                transform=self.get_transform(stage=stage, dataset=dataset),
            )

        return datasets


class CustomSupervisedConfigRunner(RunnerMixin, SupervisedConfigRunner):
    pass


if IS_HYDRA_AVAILABLE:
    from catalyst.dl import SupervisedHydraRunner

    class CustomSupervisedHydraRunner(RunnerMixin, SupervisedHydraRunner):
        pass

    __all__ = ["CustomSupervisedConfigRunner", "CustomSupervisedHydraRunner"]
else:
    __all__ = ["CustomSupervisedConfigRunner"]
