# flake8: noqa
from collections import OrderedDict

from torchvision import transforms

from catalyst.dl import ConfigExperiment
from .dataset import SomeDataset


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):

        # CHANGE ME
        result = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return result

    def get_datasets(
        self, stage: str, batch_size: int, num_workers: int, **kwargs
    ):
        datasets = OrderedDict()

        # CHANGE TO YOUR DATASET
        trainset = SomeDataset()

        # CHANGE TO YOUR DATASET
        validset = SomeDataset()

        datasets["train"] = trainset
        datasets["valid"] = validset

        return datasets
