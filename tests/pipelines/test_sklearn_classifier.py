# flake8: noqa
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from catalyst import data, dl
from catalyst.contrib import nn
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

TRAIN_EPOCH = 5
LR = 0.001
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
        # 1. generate data
        num_samples, num_features, num_classes = int(1e4), int(30), 3
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features,
            n_repeated=0,
            n_redundant=0,
            n_classes=num_classes,
            n_clusters_per_class=1,
        )
        X, y = torch.tensor(X), torch.tensor(y)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, num_workers=1, shuffle=True)

        # 2. model, optimizer and scheduler
        hidden_size, out_features = 20, 16
        model = nn.Sequential(
            nn.Linear(num_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_features)
        )
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # 3. criterion with triplets sampling
        sampler_inbatch = data.HardTripletsSampler(norm_required=False)
        criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

        # 4. training with catalyst Runner
        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch) -> None:
                features, targets = batch["features"].float(), batch["targets"].long()
                embeddings = self.model(features)
                self.batch = {
                    "embeddings": embeddings,
                    "targets": targets,
                }

        callbacks = [
            dl.SklearnModelCallback(
                feature_key="embeddings",
                target_key="targets",
                train_loader="train",
                valid_loaders="valid",
                model_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                random_state=RANDOM_STATE,
                n_estimators=100,
            ),
            dl.ControlFlowCallback(
                dl.AccuracyCallback(
                    target_key="targets", input_key="sklearn_predict", topk_args=(1, 3)
                ),
                loaders="valid",
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine or dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            scheduler=scheduler,
            loaders={"train": loader, "valid": loader},
            verbose=False,
            valid_loader="valid",
            valid_metric="accuracy",
            minimize_valid_metric=False,
            num_epochs=TRAIN_EPOCH,
            logdir=logdir,
        )

        valid_path = Path(logdir) / "logs/valid.csv"
        best_accuracy = max(float(row["accuracy"]) for row in read_csv(valid_path))

        assert best_accuracy > 0.9


@mark.skipif(not SETTINGS.ml_required, reason="catalyst[ml] required")
def test_on_cpu():
    train_experiment("cpu")
