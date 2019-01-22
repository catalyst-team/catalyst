import collections
import torch
import torchvision
from torchvision import transforms
from catalyst.dl.datasource import AbstractDataSource


class DataSource(AbstractDataSource):
    @staticmethod
    def prepare_transforms(*, mode, stage=None, **kwargs):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]
        )

    @staticmethod
    def prepare_loaders(
        *,
        mode: str,
        stage: str = None,
        n_workers: int = None,
        batch_size: int = None,
        **kwargs
    ):
        loaders = collections.OrderedDict()

        # Only valid split for testing speed
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "./data",
                train=False,
                download=True,
                transform=DataSource.prepare_transforms(
                    mode="train", stage=stage
                )
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "./data",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.3081, ))
                    ]
                )
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers
        )

        loaders["train"] = train_loader
        loaders["valid"] = test_loader

        return loaders
