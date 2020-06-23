# flake8: noqa
from collections import OrderedDict

import torch
import torchvision

from catalyst import utils
from catalyst.dl import ConfigExperiment


class CIFAR10(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."""

    def __getitem__(self, index: int):
        """Fetch a data sample for a given index.

        Args:
            index (int): index of the element in the dataset

        Returns:
            Single element by index
        """
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform({"image": image})["image"]

        return image, target


class Experiment(ConfigExperiment):
    """``ConfigExperiment`` with CIFAR10 dataset."""

    def get_model(self, stage: str):
        """
        Model specification for current stage

        Args:
            stage: current stage name

        Returns:
            model
        """
        model = super().get_model(stage=stage)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        conv_layres = ["conv1", "pool", "conv2"]

        if stage == "stage2":
            for key in conv_layres:
                layer = getattr(model, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model

    def get_datasets(self, stage: str, **kwargs):
        """Provides train/validation subsets from CIFAR10 dataset.

        Args:
            stage (str): stage name e.g. ``'stage1'`` or ``'infer'``
        """
        datasets = OrderedDict()
        for mode in ("train", "valid"):
            datasets[mode] = CIFAR10(
                root="./data",
                train=(mode == "train"),
                download=True,
                transform=self.get_transforms(stage=stage, dataset=mode),
            )

        return datasets
