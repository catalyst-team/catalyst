# flake8: noqa
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.optim import Adam

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.models import MnistSimpleNet
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.data import Compose, Normalize, ToTensor
from catalyst.data.dataset import ContrastiveDataset
from catalyst.settings import SETTINGS


def read_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                colnames = row
            else:
                yield {colname: val for colname, val in zip(colnames, row)}


BATCH_SIZE = 1024
TRAIN_EPOCH = 5
LR = 0.01
RANDOM_STATE = 42

if SETTINGS.ml_required:
    from sklearn.ensemble import RandomForestClassifier

if SETTINGS.cv_required:
    import torchvision


def train_experiment(device, engine=None):

    with TemporaryDirectory() as logdir:

        transforms = Compose(
            [   
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )
        mnist = MNIST("./logdir", train=True, download=True, transform=None)
        contrastive_mnist = ContrastiveDataset(mnist, transforms=transforms)

        train_loader = torch.utils.data.DataLoader(contrastive_mnist, batch_size=BATCH_SIZE)

        # 2. model and optimizer
        encoder = MnistSimpleNet(out_features=16)
        projection_head = nn.Sequential(
            nn.Linear(16, 16, bias=False), nn.ReLU(inplace=True), nn.Linear(16, 16, bias=True),
        )

        class ContrastiveModel(torch.nn.Module):
            def __init__(self, model, encoder):
                super(ContrastiveModel, self).__init__()
                self.model = model
                self.encoder = encoder

            def forward(self, x):
                emb = self.encoder(x)
                projection = self.model(emb)
                return emb, projection

        model = ContrastiveModel(model=projection_head, encoder=encoder)

        optimizer = Adam(model.parameters(), lr=LR)

        # 3. criterion with triplets sampling
        criterion = NTXentLoss(tau=0.1)

        callbacks = [
            dl.ControlFlowCallback(
                dl.CriterionCallback(
                    input_key="projection_left", target_key="projection_right", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.SklearnModelCallback(
                feature_key="embedding_left",
                target_key="target",
                train_loader="train",
                valid_loaders="valid",
                model_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                random_state=RANDOM_STATE,
                n_estimators=10,
            ),
            dl.ControlFlowCallback(
                dl.AccuracyCallback(
                    target_key="target", input_key="sklearn_predict", topk_args=(1, 3)
                ),
                loaders="valid",
            ),
        ]

        runner = dl.ContrastiveRunner()

        logdir = "./logdir"
        runner.train(
            model=model,
            engine=engine or dl.DeviceEngine(device),
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": train_loader},
            verbose=True,
            logdir=logdir,
            valid_loader="train",
            valid_metric="loss",
            minimize_valid_metric=True,
            num_epochs=10,
        )

        valid_path = Path(logdir) / "logs/valid.csv"
        best_accuracy = max(
            float(row["accuracy"]) for row in read_csv(valid_path) if row["accuracy"] != "accuracy"
        )

        assert best_accuracy > 0.7


@mark.skipif(not SETTINGS.ml_required or not SETTINGS.cv_required, reason="catalyst[ml] required")
def test_on_cpu():
    train_experiment("cpu")
