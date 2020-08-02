# flake8: noqa
from collections import OrderedDict

import torchvision
from torchvision import transforms

from catalyst.dl import ConfigExperiment


class SimpleExperiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        """
        @TODO: Docs. Contribution is welcome
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_datasets(self, stage: str, **kwargs):
        """
        @TODO: Docs. Contribution is welcome
        """
        datasets = OrderedDict()

        trainset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=SimpleExperiment.get_transforms(
                stage=stage, mode="train"
            ),
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=SimpleExperiment.get_transforms(
                stage=stage, mode="valid"
            ),
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
