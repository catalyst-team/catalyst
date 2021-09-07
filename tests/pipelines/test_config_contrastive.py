# flake8: noqa

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark
from torch import nn

from catalyst.data import Compose, Normalize, ToTensor
from catalyst.dl import SelfSupervisedConfigRunner
from catalyst.registry import Registry
from catalyst.settings import SETTINGS


def read_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                colnames = row
            else:
                yield {colname: val for colname, val in zip(colnames, row)}


if SETTINGS.cv_required:
    import torchvision

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )

    transform_original = transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)),])


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=False), nn.ReLU(inplace=True), nn.Linear(16, 16, bias=True),
        )

    def forward(self, x):
        return self.seq(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 16), nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        emb = self.encoder(x)
        return emb


class ContrastiveModel(nn.Module):
    def __init__(self, projection_head, encoder):
        super(ContrastiveModel, self).__init__()
        self.projection_head = projection_head
        self.encoder = encoder

    def forward(self, x):
        emb = self.encoder(x)
        projection = self.projection_head(emb)
        return emb, projection


Registry(ContrastiveModel)
Registry(ProjectionHead)
Registry(Encoder)


def train_experiment(engine):
    with TemporaryDirectory() as logdir:

        runner = SelfSupervisedConfigRunner(
            config={
                "args": {
                    "logdir": logdir,
                    "valid_loader": "valid",
                    "valid_metric": "accuracy01",
                    "minimize_valid_metric": False,
                    "verbose": True,
                },
                "model": {
                    "_target_": "ContrastiveModel",
                    "projection_head": {"_target_": "ProjectionHead"},
                    "encoder": {"_target_": "Encoder"},
                },
                "engine": engine,
                "loggers": {
                    "console": {"_target_": "ConsoleLogger"},
                    "csv": {"_target_": "CSVLogger", "logdir": logdir},
                    "tensorboard": {"_target_": "TensorboardLogger", "logdir": logdir},
                },
                "stages": {
                    "stage1": {
                        "num_epochs": 5,
                        "criterion": {"_target_": "NTXentLoss", "tau": 0.1},
                        "optimizer": {"_target_": "Adam", "lr": 1e-3},
                        "scheduler": {"_target_": "MultiStepLR", "milestones": [2]},
                        "loaders": {
                            "batch_size": 1024,
                            "datasets": {
                                "train": {
                                    "_target_": "catalyst.data.dataset.SelfSupervisedDatasetWrapper",
                                    "dataset": {
                                        "_target_": "catalyst.contrib.datasets.MNIST",
                                        "root": logdir,
                                        "train": True,
                                        "download": True,
                                    },
                                    "transforms": transforms,
                                    "transform_original": transform_original,
                                },
                                "valid": {
                                    "_target_": "catalyst.data.dataset.SelfSupervisedDatasetWrapper",
                                    "dataset": {
                                        "_target_": "catalyst.contrib.datasets.MNIST",
                                        "root": logdir,
                                        "train": False,
                                        "download": True,
                                    },
                                    "transforms": transforms,
                                    "transform_original": transform_original,
                                },
                            },
                        },
                        "callbacks": {
                            "criterion": {
                                "_target_": "CriterionCallback",
                                "input_key": "projection_left",
                                "target_key": "projection_right",
                                "metric_key": "loss",
                            },
                            "sklearn_model": {
                                "_target_": "catalyst.dl.SklearnModelCallback",
                                "feature_key": "embedding_left",
                                "target_key": "target",
                                "train_loader": "train",
                                "valid_loaders": "valid",
                                "model_fn": "ensemble.RandomForestClassifier",
                                "predict_method": "predict_proba",
                                "predict_key": "sklearn_predict",
                                "random_state": 42,
                                "n_estimators": 10,
                            },
                            "accuracy": {
                                "_target_": "catalyst.dl.ControlFlowCallback",
                                "base_callback": {
                                    "_target_": "catalyst.dl.AccuracyCallback",
                                    "target_key": "target",
                                    "input_key": "sklearn_predict",
                                    "topk_args": [1, 3],
                                },
                                "loaders": "valid",
                            },
                            "optimizer": {"_target_": "OptimizerCallback", "metric_key": "loss",},
                            "scheduler": {"_target_": "SchedulerCallback"},
                            "checkpointer": {
                                "_target_": "CheckpointCallback",
                                "logdir": logdir,
                                "loader_key": "train",
                                "metric_key": "loss",
                                "minimize": True,
                                "save_n_best": 1,
                            },
                        },
                    },
                },
            },
        )

        runner.run()

        valid_path = Path(logdir) / "logs/valid.csv"
        best_accuracy = max(
            float(row["accuracy"]) for row in read_csv(valid_path) if row["accuracy"] != "accuracy"
        )

        assert best_accuracy > 0.7


# Torch
@mark.skipif(
    not SETTINGS.ml_required or not SETTINGS.cv_required,
    reason="catalyst[ml] and catalyst[cv] required",
)
def test_classification_on_cpu():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"})
