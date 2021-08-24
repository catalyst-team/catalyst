# flake8: noqa
from common import ContrastiveRunner
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms.transforms import ColorJitter

from catalyst import data, dl
from catalyst.contrib import datasets, models, nn
from catalyst.contrib.data.datawrappers import simCLRDatasetWrapper
from catalyst.contrib.datasets.cifar import Cifar10MLDataset, CifarQGDataset
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion import NTXentLoss

batch_size = 1000
aug_strength = 1.0
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        torchvision.transforms.ColorJitter(
            aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.2
        ),
    ]
)


cifar_train = CIFAR10(root="./data", download=True, transform=None)
simCLR_train = simCLRDatasetWrapper(cifar_train, transforms=transforms)
train_loader = torch.utils.data.DataLoader(simCLR_train, batch_size=batch_size, num_workers=5)

# cifar_test = CifarQGDataset(root="./data", download=True)
# valid_loader = torch.utils.data.DataLoader(
#     simCLRDatasetWrapper(cifar_test, transforms=transforms), batch_size=batch_size, num_workers=5
# )


class Model(nn.Module):
    def __init__(self, feature_dim=128, **resnet_kwargs):
        super(Model, self).__init__()
        # encoder
        self.encoder = nn.Sequential(ResnetEncoder(**resnet_kwargs), nn.Flatten())
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        feature = self.encoder(x)
        out = self.g(feature)
        return F.normalize(out, dim=-1)


model = Model(feature_dim=256, arch="resnet50", frozen=False,)


# 2. model and optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# 3. criterion with triplets sampling
criterion = NTXentLoss(tau=0.1)

print(list(model.parameters())[0][0][0])
callbacks = [
    dl.ControlFlowCallback(
        dl.CriterionCallback(input_key="proj1", target_key="proj2", metric_key="loss"),
        loaders="train",
    )
]

runner = ContrastiveRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders={
        "train": train_loader,
        # "valid": valid_loader
    },
    verbose=True,
    logdir="./logs",
    valid_loader="train",
    valid_metric="loss",
    minimize_valid_metric=True,
    num_epochs=100,
)

print(list(model.parameters())[0][0][0])
