# flake8: noqa

import collections
from copy import deepcopy
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import Sampler

from catalyst import dl
from catalyst.registry import REGISTRY
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

NUM_SAMPLES, FEATURES_SHAPE, NUM_CLASSES = int(1e4), 16, 4


class CustomTensorDataset(TensorDataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform: "Callable" = None):
        super().__init__(X, y)
        self.transform = transform

    def __getitem__(self, index: int) -> "Tuple[torch.Tensor, torch.Tensor]":
        X, y = super().__getitem__(index)

        if self.transform is not None:
            X = self.transform(X)
        X = torch.flatten(X)

        return X, y


class CustomMiniEpochSampler(Sampler):
    def __init__(self, custom_sampler_key: Sampler):
        super().__init__(None)
        self._sampler = custom_sampler_key

    def shuffle(self) -> None:
        self._sampler.shuffle()

    def __iter__(self) -> "Iterator[int]":
        return self._sampler.__iter__()

    def __len__(self) -> int:
        return self._sampler.__len__()


def train_experiment(engine, test_transform: bool = False, test_sampler: bool = False):
    with TemporaryDirectory() as logdir:
        REGISTRY.add(CustomTensorDataset)
        REGISTRY.add(CustomMiniEpochSampler)

        X = torch.rand(NUM_SAMPLES, 1, FEATURES_SHAPE, FEATURES_SHAPE)
        y = (torch.rand(NUM_SAMPLES,) * NUM_CLASSES).to(torch.int64)

        if test_transform:
            normalize_transform = {"_target_": "transform.Normalize", "mean": [0], "std": [1]}

            # use `Compose` to test complex transforms support (`_transforms_` keyword)
            transform_params = {
                "_target_": "transform.Compose",
                "_transforms_": ("transforms",),
                "transforms": [normalize_transform],
            }
        else:
            transform_params = None

        if test_sampler:
            base_sampler_params = {
                "_target_": "MiniEpochSampler",
                "data_len": X.shape[0],
                "mini_epoch_len": X.shape[0] // 10,
                "drop_last": True,
                "shuffle": "per_epoch",
            }

            # use `CustomMiniEpochSampler` to test `BatchSampler`-like samplers support
            #  (`_samplers_` keyword in sampler params)
            sampler_params = {
                "train": {
                    "_target_": "CustomMiniEpochSampler",
                    "_samplers_": ("custom_sampler_key",),
                    "custom_sampler_key": base_sampler_params,
                }
            }
        else:
            sampler_params = {}

        runner = dl.SupervisedConfigRunner(
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
                    "in_features": FEATURES_SHAPE * FEATURES_SHAPE,
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
                                "train": {
                                    "_target_": "CustomTensorDataset",
                                    "X": X,
                                    "y": y,
                                    "transform": transform_params,
                                },
                                "valid": {
                                    "_target_": "CustomTensorDataset",
                                    "X": X,
                                    "y": y,
                                    "transform": transform_params,
                                },
                            },
                            "samplers": sampler_params,
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


def test_base_experiment():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"})


def test_get_transforms():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"}, test_transform=True)


def test_get_samplers():
    train_experiment({"_target_": "DeviceEngine", "device": "cpu"}, test_sampler=True)
