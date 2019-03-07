from collections import OrderedDict
import torchvision
from torchvision import transforms
from catalyst.dl.experiments import ConfigExperiment


class Experiment(ConfigExperiment):

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=Experiment.get_transforms(
                mode="train", stage=stage
            )
        )
        testset = torchvision.datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=Experiment.get_transforms(
                mode="train", stage=stage
            )
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
