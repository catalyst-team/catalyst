# flake8: noqa
from collections import OrderedDict

from torch.utils.data import Subset

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

    def get_datasets(
        self,
        stage: str,
        n_samples: int = 320,
        duplicate_loaders: bool = False,
        **kwargs
    ):
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
            if n_samples > 0:
                trainset = Subset(trainset, list(range(n_samples)))
                testset = Subset(testset, list(range(n_samples)))
            datasets["train"] = trainset
            datasets["valid"] = testset
            if duplicate_loaders:
                datasets["train_additional"] = trainset
                datasets["valid_additional"] = testset
        else:
            testset = MNIST(
                "./data",
                train=False,
                download=True,
                transform=Experiment.get_transforms(stage=stage, mode="valid"),
            )
            if n_samples > 0:
                testset = Subset(testset, list(range(n_samples)))
            datasets["infer"] = testset

        return datasets
