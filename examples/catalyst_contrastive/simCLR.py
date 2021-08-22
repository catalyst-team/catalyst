# flake8: noqa
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import ColorJitter, Lambda, RandomCrop, ToPILImage, ToTensor

from catalyst import data, dl
from catalyst.contrib import datasets, models, nn
from catalyst.contrib.data.datawrappers import simCLRDatasetWrapper
from catalyst.contrib.datasets.cifar import Cifar10MLDataset, CifarQGDataset
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.data.transforms import Compose, Normalize, ToTensor

batch_size = 10

transforms = Compose(
    [
        #
        ToPILImage(),
        RandomCrop(26),
        ColorJitter(),
        ToTensor(),
        Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)


cifar_train = Cifar10MLDataset(root="./data", download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(
    simCLRDatasetWrapper(cifar_train, transforms=transforms), batch_size=batch_size, num_workers=5
)
cifar_test = CifarQGDataset(root="./data", download=True)
valid_loader = torch.utils.data.DataLoader(
    simCLRDatasetWrapper(cifar_test, transforms=transforms), batch_size=batch_size, num_workers=5
)


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


# 4. training with catalyst Runner
class CustomRunner(dl.Runner):
    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        emb1 = self.model(batch["image_aug1"].to(self.device))
        emb2 = self.model(batch["image_aug2"].to(self.device))
        self.batch = {"proj1": emb1, "proj2": emb2}


callbacks = [
    dl.ControlFlowCallback(
        dl.CriterionCallback(input_key="proj1", target_key="proj2", metric_key="loss"),
        loaders="train",
    )
]

runner = CustomRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders={"train": train_loader, "valid": valid_loader},
    verbose=False,
    logdir="./logs",
    valid_loader="train",
    valid_metric="loss",
    minimize_valid_metric=True,
    num_epochs=20,
)
