from collections import OrderedDict

import torchvision
from catalyst.dl import ConfigExperiment
from torchvision import transforms


# data loaders & transforms
class MNISTGANExperiment(ConfigExperiment):
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
            transform=MNISTGANExperiment.get_transforms(
                stage=stage, mode="train"
            )
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=MNISTGANExperiment.get_transforms(
                stage=stage, mode="valid"
            )
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
