from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from catalyst import utils
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage2":
            for key in ["conv1", "pool", "conv2"]:
                layer = getattr(model_, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model_

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="train")
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="valid")
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
