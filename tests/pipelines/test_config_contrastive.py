# flake8: noqa

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

from torch import nn

from catalyst.data import Compose, Normalize, ToTensor
from catalyst.dl import SelfSupervisedConfigRunner
from catalyst.registry import Registry
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

if SETTINGS.cv_required:
    import torchvision

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


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=False), nn.ReLU(inplace=True), nn.Linear(16, 16, bias=True)
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
                    "verbose": False,
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
                        "num_epochs": 2,
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
                            "optimizer": {"_target_": "OptimizerCallback", "metric_key": "loss"},
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

        metrics_path = Path(logdir) / "_metrics.json"
        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        assert metrics["best"]["valid"]["accuracy"] > 0.6


requirements_satisfied = SETTINGS.ml_required and SETTINGS.cv_required

# Torch
@mark.skipif(not requirements_satisfied, reason="catalyst[ml] and catalyst[cv] required")
def test_contrastive_on_cpu():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"})


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE]),
    reason="catalyst[ml], catalyst[cv] and CUDA device are required",
)
def test_classification_on_torch_cuda0():
    train_experiment({"_target_": "DeviceEngine", "device": "cuda:0"})


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found or requriments are not satisfied",
)
def test_classification_on_torch_cuda1():
    train_experiment({"_target_": "DeviceEngine", "device": "cuda:1"})


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found or requriments are not satisfied",
)
def test_classification_on_torch_dp():
    train_experiment({"_target_": "DataParallelEngine"})


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found or requriments are not satisfied",
)
def test_classification_on_torch_ddp():
    train_experiment({"_target_": "DistributedDataParallelEngine"})


# AMP
@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found or requriments are not satisfied",
)
def test_classification_on_amp():
    train_experiment({"_target_": "AMPEngine"})


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]
    ),
    reason="No CUDA>=2 or AMP found or requriments are not satisfied",
)
def test_classification_on_amp_dp():
    train_experiment({"_target_": "DataParallelAMPEngine"})


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]
    ),
    reason="No CUDA>=2 or AMP found or requriments are not satisfied",
)
def test_classification_on_amp_ddp():
    train_experiment({"_target_": "DistributedDataParallelAMPEngine"})


# APEX
@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE, SETTINGS.apex_required]),
    reason="No CUDA or Apex found or requriments are not satisfied",
)
def test_classification_on_apex():
    train_experiment({"_target_": "APEXEngine"})


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.apex_required]
    ),
    reason="No CUDA>=2 or Apex found or requriments are not satisfied",
)
def test_classification_on_apex_dp():
    train_experiment({"_target_": "DataParallelApexEngine"})


@mark.skipif(
    not all(
        [requirements_satisfied, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.apex_required]
    ),
    reason="No CUDA>=2 or Apex found or requriments are not satisfied",
)
def test_classification_on_apex_ddp():
    train_experiment({"_target_": "DistributedDataParallelApexEngine"})
