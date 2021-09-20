# flake8: noqa
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import data, dl
from catalyst.contrib import datasets, models, nn
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import Compose, Normalize, ToTensor
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from sklearn.ensemble import RandomForestClassifier

TRAIN_EPOCH = 1
LR = 0.01
RANDOM_STATE = 42


def read_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                colnames = row
            else:
                yield {colname: val for colname, val in zip(colnames, row)}


def train_experiment(device, engine=None):
    with TemporaryDirectory() as logdir:
        from catalyst import utils

        utils.set_global_seed(RANDOM_STATE)
        # 1. train, valid and test loaders
        transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        train_data = MNIST(os.getcwd(), train=True, download=True, transform=transforms)
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = data.BatchBalanceClassSampler(train_labels, num_classes=10, num_samples=4)
        train_loader = DataLoader(train_data, batch_sampler=train_sampler)

        valid_dataset = MNIST(root=os.getcwd(), transform=transforms, train=False, download=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=32)

        test_dataset = MNIST(root=os.getcwd(), transform=transforms, train=False, download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32)

        # 2. model and optimizer
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 16), nn.LeakyReLU(inplace=True))
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # 3. criterion with triplets sampling
        sampler_inbatch = data.HardTripletsSampler(norm_required=False)
        criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

        # 4. training with catalyst Runner
        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch) -> None:
                images, targets = batch["features"].float(), batch["targets"].long()
                features = self.model(images)
                self.batch = {
                    "embeddings": features,
                    "targets": targets,
                }

        callbacks = [
            dl.ControlFlowCallback(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.SklearnModelCallback(
                feature_key="embeddings",
                target_key="targets",
                train_loader="train",
                valid_loaders=["valid", "infer"],
                model_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                random_state=RANDOM_STATE,
                n_estimators=50,
            ),
            dl.ControlFlowCallback(
                dl.AccuracyCallback(
                    target_key="targets", input_key="sklearn_predict", topk_args=(1, 3)
                ),
                loaders=["valid", "infer"],
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine or dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": valid_loader, "infer": test_loader},
            verbose=False,
            valid_loader="valid",
            valid_metric="accuracy",
            minimize_valid_metric=False,
            num_epochs=TRAIN_EPOCH,
            logdir=logdir,
        )

        valid_path = Path(logdir) / "logs/infer.csv"
        best_accuracy = max(float(row["accuracy"]) for row in read_csv(valid_path))

        assert best_accuracy > 0.8


@mark.skipif(not SETTINGS.ml_required, reason="catalyst[ml] required")
def test_on_cpu():
    train_experiment("cpu")
