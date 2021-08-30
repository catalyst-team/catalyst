import csv
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.data.datawrappers import ContrastiveDataset
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.criterion import NTXentLoss


def read_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                colnames = row
            else:
                yield {colname: val for colname, val in zip(colnames, row)}


batch_size = 1024
aug_strength = 1.0

transforms = transform = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
mnist = MNIST("./logdir", train=True, download=True, transform=None)
contrastive_mnist = ContrastiveDataset(mnist, transforms=transforms)

# Cifar10MLDataset has mistakes
# cifar_train = Cifar10MLDataset(root="./data", download=True, transform=None)


train_loader = torch.utils.data.DataLoader(contrastive_mnist, batch_size=batch_size, num_workers=2)

# cifar_test = CifarQGDataset(root="./data", download=True)
# valid_loader = torch.utils.data.DataLoader(
#     simCLRDatasetWrapper(cifar_test, transforms=transforms), batch_size=batch_size, num_workers=5
# )


class Model(nn.Module):
    def __init__(self, feature_dim=128, **resnet_kwargs):
        super(Model, self).__init__()
        # encoder
        inner_size = 28 * 28
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(),
            nn.Linear(inner_size, inner_size),
        )
        # projection head
        self.g = nn.Sequential(
            nn.Linear(inner_size, inner_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inner_size, feature_dim, bias=True),
        )

    def forward(self, x):
        feature = self.encoder(x)
        out = self.g(feature)
        return F.normalize(out, dim=-1)


model = Model(feature_dim=30, arch="resnet50", frozen=False,)

# 2. model and optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# 3. criterion with triplets sampling
criterion = NTXentLoss(tau=0.1)

callbacks = [
    dl.ControlFlowCallback(
        dl.CriterionCallback(input_key="proj1", target_key="proj2", metric_key="loss"),
        loaders="train",
    ),
    dl.SklearnModelCallback(
        feature_key="proj1",
        target_key="target",
        train_loader="train",
        valid_loaders="valid",
        model_fn=RandomForestClassifier,
        predict_method="predict_proba",
        predict_key="sklearn_predict",
        random_state=4545,
        n_estimators=10,
    ),
    dl.ControlFlowCallback(
        dl.AccuracyCallback(target_key="target", input_key="sklearn_predict", topk_args=(1, 3)),
        loaders="valid",
    ),
]

runner = dl.ContrastiveRunner()

logdir = "./logdir"
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders={"train": train_loader, "valid": train_loader},
    verbose=True,
    logdir=logdir,
    valid_loader="train",
    valid_metric="loss",
    minimize_valid_metric=True,
    num_epochs=3,
)

valid_path = Path(logdir) / "logs/valid.csv"
best_accuracy = max(
    float(row["accuracy"]) for row in read_csv(valid_path) if row["accuracy"] != "accuracy"
)

assert best_accuracy > 0.7
