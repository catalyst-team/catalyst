# flake8: noqa
from collections import OrderedDict
import os

from torch.utils.data import Subset

from catalyst.contrib.datasets import MNIST
from catalyst.data.cv import Compose, Normalize, ToTensor
from catalyst.dl import ConfigExperiment
from catalyst import dl, registry


@registry.Callback
class KeyboardInteruptCallback(dl.Callback):
    def __init__(self, interrupt_epoch: int = None):
        super().__init__(dl.CallbackOrder.Metric)
        self.interrupt_epoch = interrupt_epoch

    def on_batch_end(self, runner: dl.IRunner):
        if self.interrupt_epoch is None:
            interrupt_at = int(os.environ.get("INTERRUPT_EPOCH", -1))
        else:
            interrupt_at = self.interrupt_epoch
        if runner.epoch == interrupt_at:
            raise KeyboardInterrupt()


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

    def get_datasets(self, stage: str, n_samples: int = 320, **kwargs):
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
