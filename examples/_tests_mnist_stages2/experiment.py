from collections import OrderedDict

import torchvision

from catalyst.dl.experiment import ConfigExperiment


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int):
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform({"image": image})["image"]

        return image, target


class Experiment(ConfigExperiment):
    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        for mode in ("train", "valid"):
            datasets[mode] = torchvision.datasets.MNIST(
                "./data",
                train=False,
                download=True,
                transform=self.get_transforms(mode=mode, stage=stage)
            )

        return datasets
