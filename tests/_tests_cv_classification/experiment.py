# flake8: noqa
from collections import OrderedDict

from catalyst.contrib.datasets import MNIST
from catalyst.data.cv import Compose, Normalize, ToTensor
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        """
        @TODO: Docs. Contribution is welcome
        """
        return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def get_datasets(self, stage: str, **kwargs):
        """
        @TODO: Docs. Contribution is welcome
        """
        datasets = OrderedDict()

        if stage != "infer":
            trainset = MNIST(
                "./data",
                train=False,
                download=True,
                transform=Experiment.get_transforms(stage=stage, mode="train"),
            )
            testset = MNIST(
                "./data",
                train=False,
                download=True,
                transform=Experiment.get_transforms(stage=stage, mode="valid"),
            )

            datasets["train"] = trainset
            datasets["valid"] = testset
        else:
            testset = MNIST(
                "./data",
                train=False,
                download=True,
                transform=Experiment.get_transforms(stage=stage, mode="valid"),
            )
            datasets["infer"] = testset

        return datasets
