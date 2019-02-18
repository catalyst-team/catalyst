from collections import OrderedDict

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from catalyst.dl import ConfigExperiment


class DictDatasetAdapter(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, target = self.dataset[index]

        return dict(image=image, targets=target)

    def __len__(self):
        return len(self.dataset)


class Experiment(ConfigExperiment):

    def get_transforms(self, mode, stage: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=self.get_transforms('train', stage)
        )
        train_loader = DataLoader(
            DictDatasetAdapter(train_set),
            batch_size=100,
            shuffle=True,
            num_workers=3
        )

        validation_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=self.get_transforms('valid', stage)
        )

        validation_loader = DataLoader(
            DictDatasetAdapter(validation_set),
            batch_size=100,
            shuffle=True,
            num_workers=3
        )

        return OrderedDict(train=train_loader, valid=validation_loader)
