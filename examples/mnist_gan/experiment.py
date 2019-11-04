from collections import OrderedDict

import torchvision
from torchvision import transforms

from catalyst.dl import ConfigExperiment


# data loaders & transforms
class MnistGanExperiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))]
        )

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=MnistGanExperiment.get_transforms(
                stage=stage, mode="train"
            )
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=MnistGanExperiment.get_transforms(
                stage=stage, mode="valid"
            )
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
