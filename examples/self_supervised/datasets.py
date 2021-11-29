from PIL import Image

import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, STL10

datasets = {
    "CIFAR-10": {
        "dataset": CIFAR10,
        "train_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                torchvision.transforms.RandomGrayscale(p=0.1),
                # torchvision.transforms.RandomResizedCrop(
                #     64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC,
                # ),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282)),
            ]
        ),
        "valid_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
    },
    "CIFAR-100": {
        "dataset": CIFAR100,
        "train_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                torchvision.transforms.RandomGrayscale(p=0.1),
                # torchvision.transforms.RandomResizedCrop(
                #     64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC,
                # ),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282)),
            ]
        ),
        "valid_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
    },
    "STL10": {
        "dataset": STL10,
        "train_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                torchvision.transforms.RandomGrayscale(p=0.1),
                torchvision.transforms.RandomCrop(32),
                # torchvision.transforms.RandomResizedCrop(
                #     64, scale=(0.2, 1.0), ratio=(0.75, (4 / 3)), interpolation=Image.BICUBIC,
                # ),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
            ]
        ),
        "valid_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(70, interpolation=Image.BICUBIC),
                torchvision.transforms.CenterCrop(64),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
            ]
        ),
    },
}
