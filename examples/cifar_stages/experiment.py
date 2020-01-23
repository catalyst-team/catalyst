from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

from catalyst import utils
from catalyst.dl import ConfigExperiment


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int):
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform({"image": image})["image"]

        return image, target


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

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        for mode in ("train", "valid"):
            datasets[mode] = CIFAR10(
                root="./data",
                train=(mode == "train"),
                download=True,
                transform=self.get_transforms(stage=stage, dataset=mode),
            )

        return datasets
