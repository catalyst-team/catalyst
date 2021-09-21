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
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.data import Compose, Normalize, ToTensor
from catalyst.data.dataset import SelfSupervisedDatasetWrapper
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


def read_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                colnames = row
            else:
                yield {colname: val for colname, val in zip(colnames, row)}


BATCH_SIZE = 1024
TRAIN_EPOCH = 2
LR = 0.01
RANDOM_STATE = 42

if SETTINGS.ml_required:
    from sklearn.ensemble import RandomForestClassifier

if SETTINGS.cv_required:
    import torchvision


def train_experiment(device, engine=None):

    with TemporaryDirectory() as logdir:

        # 1. data and transforms

        transforms = Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop((28, 28)),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )

        transform_original = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        mnist = MNIST("./logdir", train=True, download=True, transform=None)
        contrastive_mnist = SelfSupervisedDatasetWrapper(
            mnist, transforms=transforms, transform_original=transform_original
        )
        train_loader = torch.utils.data.DataLoader(contrastive_mnist, batch_size=BATCH_SIZE)

        mnist_valid = MNIST("./logdir", train=False, download=True, transform=None)
        contrastive_valid = SelfSupervisedDatasetWrapper(
            mnist_valid, transforms=transforms, transform_original=transform_original
        )
        valid_loader = torch.utils.data.DataLoader(contrastive_valid, batch_size=BATCH_SIZE)

        # 2. model and optimizer
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 16), nn.LeakyReLU(inplace=True))
        projection_head = nn.Sequential(
            nn.Linear(16, 16, bias=False), nn.ReLU(inplace=True), nn.Linear(16, 16, bias=True)
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
                n_estimators=50,
            ),
            dl.ControlFlowCallback(
                dl.AccuracyCallback(
                    target_key="target", input_key="sklearn_predict", topk_args=(1, 3)
                ),
                loaders="valid",
            ),
        ]

        runner = dl.SelfSupervisedRunner()

        logdir = "./logdir"
        runner.train(
            model=model,
            engine=engine or dl.DeviceEngine(device),
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": valid_loader},
            verbose=False,
            logdir=logdir,
            valid_loader="train",
            valid_metric="loss",
            minimize_valid_metric=True,
            num_epochs=TRAIN_EPOCH,
        )

        valid_path = Path(logdir) / "logs/valid.csv"
        best_accuracy = max(
            float(row["accuracy"]) for row in read_csv(valid_path) if row["accuracy"] != "accuracy"
        )

        assert best_accuracy > 0.6


requirements_satisfied = SETTINGS.ml_required and SETTINGS.cv_required


@mark.skipif(not requirements_satisfied, reason="catalyst[ml] and catalyst[cv] required")
def test_on_cpu():
    train_experiment("cpu")


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE]),
    reason="catalyst[ml], catalyst[cv] and CUDA device are required",
)
def test_on_torch_cuda0():
    train_experiment("cuda:0")


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found or requriments are not satisfied",
)
def test_on_torch_cuda1():
    train_experiment("cuda:1")


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found or requriments are not satisfied",
)
def test_on_torch_dp():
    train_experiment(None, dl.DataParallelEngine())


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_on_torch_ddp():
    train_experiment(None, dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found or requriments are not satisfied",
)
def test_on_amp():
    train_experiment(None, dl.AMPEngine())


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]
    ),
    reason="No CUDA>=2 or AMP found or requriments are not satisfied",
)
def test_on_amp_dp():
    train_experiment(None, dl.DataParallelAMPEngine())


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]
    ),
    reason="No CUDA>=2 or AMP found or requriments are not satisfied",
)
def test_on_amp_ddp():
    train_experiment(None, dl.DistributedDataParallelAMPEngine())


# APEX
@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, SETTINGS.apex_required]),
    reason="No CUDA or Apex found or requriments are not satisfied",
)
def test_on_apex():
    train_experiment(None, dl.APEXEngine())


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.apex_required]
    ),
    reason="No CUDA>=2 or Apex found or requriments are not satisfied",
)
def test_on_apex_dp():
    train_experiment(None, dl.DataParallelAPEXEngine())


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.apex_required]
    ),
    reason="No CUDA>=2 or Apex found or requriments are not satisfied",
)
def test_on_apex_ddp():
    train_experiment(None, dl.DistributedDataParallelAPEXEngine())
