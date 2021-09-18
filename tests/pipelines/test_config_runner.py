# flake8: noqa

import collections
from copy import deepcopy
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch.utils.data import TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES = int(1e4), int(1e1), 4


class CustomConfigRunner(dl.SupervisedConfigRunner):
    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        params = deepcopy(self._stage_config[stage]["loaders"]["datasets"])
        num_samples = params.get("num_samples", NUM_SAMPLES)
        num_features = params.get("num_features", NUM_FEATURES)
        num_classes = params.get("num_classes", NUM_CLASSES)

        # sample data
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples) * num_classes).to(torch.int64)

        # pytorch dataset
        dataset = TensorDataset(X, y)
        datasets = {"train": dataset, "valid": dataset}
        return datasets


def train_experiment(engine):
    with TemporaryDirectory() as logdir:

        runner = CustomConfigRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss",
            config={
                "args": {
                    "logdir": logdir,
                    "valid_loader": "valid",
                    "valid_metric": "accuracy01",
                    "minimize_valid_metric": False,
                    "verbose": False,
                },
                "model": {
                    "_target_": "Linear",
                    "in_features": NUM_FEATURES,
                    "out_features": NUM_CLASSES,
                },
                "engine": engine,
                "loggers": {
                    "console": {"_target_": "ConsoleLogger"},
                    "csv": {"_target_": "CSVLogger", "logdir": logdir},
                    "tensorboard": {"_target_": "TensorboardLogger", "logdir": logdir},
                },
                "stages": {
                    "stage1": {
                        "num_epochs": 10,
                        "criterion": {"_target_": "CrossEntropyLoss"},
                        "optimizer": {"_target_": "Adam", "lr": 1e-3},
                        "scheduler": {"_target_": "MultiStepLR", "milestones": [2]},
                        "loaders": {
                            "batch_size": 32,
                            "num_workers": 1,
                            "datasets": {
                                "num_samples": NUM_SAMPLES,
                                "num_features": NUM_FEATURES,
                                "num_classes": NUM_CLASSES,
                            },
                        },
                        "callbacks": {
                            "accuracy": {
                                "_target_": "AccuracyCallback",
                                "input_key": "logits",
                                "target_key": "targets",
                                "num_classes": NUM_CLASSES,
                            },
                            "classification": {
                                "_target_": "PrecisionRecallF1SupportCallback",
                                "input_key": "logits",
                                "target_key": "targets",
                                "num_classes": NUM_CLASSES,
                            },
                            "criterion": {
                                "_target_": "CriterionCallback",
                                "input_key": "logits",
                                "target_key": "targets",
                                "metric_key": "loss",
                            },
                            "optimizer": {"_target_": "OptimizerCallback", "metric_key": "loss"},
                            "scheduler": {"_target_": "SchedulerCallback"},
                            "checkpointer": {
                                "_target_": "CheckpointCallback",
                                "logdir": logdir,
                                "loader_key": "valid",
                                "metric_key": "accuracy01",
                                "minimize": False,
                                "save_n_best": 3,
                            },
                        },
                    },
                },
            },
        )
        runner.run()


# Torch
def test_classification_on_cpu():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"})


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_classification_on_torch_cuda0():
    train_experiment({"_target_": "DeviceEngine", "device": "cuda:0"})


@mark.skipif(not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found")
def test_classification_on_torch_cuda1():
    train_experiment({"_target_": "DeviceEngine", "device": "cuda:1"})


@mark.skipif(not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found")
def test_classification_on_torch_dp():
    train_experiment({"_target_": "DataParallelEngine"})


@mark.skipif(not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found")
def test_classification_on_torch_ddp():
    train_experiment({"_target_": "DistributedDataParallelEngine"})


# AMP
@mark.skipif(not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found")
def test_classification_on_amp():
    train_experiment({"_target_": "AMPEngine"})


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_dp():
    train_experiment({"_target_": "DataParallelAMPEngine"})


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_ddp():
    train_experiment({"_target_": "DistributedDataParallelAMPEngine"})


# APEX
@mark.skipif(not (IS_CUDA_AVAILABLE and SETTINGS.apex_required), reason="No CUDA or Apex found")
def test_classification_on_apex():
    train_experiment({"_target_": "APEXEngine"})


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_classification_on_apex_dp():
    train_experiment({"_target_": "DataParallelApexEngine"})


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_classification_on_apex_ddp():
    train_experiment({"_target_": "DistributedDataParallelApexEngine"})
