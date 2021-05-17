# flake8: noqa

from typing import Dict
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst.callbacks import CheckpointCallback, CriterionCallback, OptimizerCallback
from catalyst.core.runner import IRunner
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

from .misc import (
    DeviceCheckCallback,
    DummyDataset,
    DummyModel,
    LossMinimizationCallback,
    ModuleTypeChecker,
    TensorTypeChecker,
)

if SETTINGS.amp_required:
    from catalyst.engines.amp import AMPEngine

logger = logging.getLogger(__name__)


class CustomRunner(IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return AMPEngine(self._device)

    def get_callbacks(self, stage: str):
        return {
            "criterion": CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            "test_nn_module": ModuleTypeChecker(),
            "test_device": DeviceCheckCallback(self._device, logger=logger),
            "test_loss_minimization": LossMinimizationCallback("loss", logger=logger),
            "test_logits_type": TensorTypeChecker("logits"),
            # "loss_type_checker": TensorTypeChecker("loss", True),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 3

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        dataset = DummyDataset(6)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return DummyModel(4, 2)

    def get_criterion(self, stage: str):
        return torch.nn.MSELoss()

    def get_optimizer(self, model, stage: str):
        return torch.optim.Adam(model.parameters())

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_trial(self):
        return None

    def get_loggers(self):
        return {"console": ConsoleLogger(), "csv": CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


def train_from_runner(device):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device)
        runner.run()


def train_from_config(device):
    with TemporaryDirectory() as logdir:
        dataset = DummyDataset(6)
        runner = SupervisedConfigRunner(
            config={
                "args": {"logdir": logdir},
                "model": {"_target_": "DummyModel", "in_features": 4, "out_features": 2},
                "engine": {"_target_": "AMPEngine", "device": device},
                "args": {"logdir": logdir},
                "stages": {
                    "stage1": {
                        "num_epochs": 10,
                        "criterion": {"_target_": "MSELoss"},
                        "optimizer": {"_target_": "Adam", "lr": 1e-3},
                        "loaders": {"batch_size": 4, "num_workers": 0},
                        "callbacks": {
                            "criterion": {
                                "_target_": "CriterionCallback",
                                "metric_key": "loss",
                                "input_key": "logits",
                                "target_key": "targets",
                            },
                            "optimizer": {"_target_": "OptimizerCallback", "metric_key": "loss"},
                            "test_nn_module": {"_target_": "ModuleTypeChecker"},
                            "test_device": {
                                "_target_": "DeviceCheckCallback",
                                "assert_device": device,
                            },
                            "test_loss_minimization": {
                                "_target_": "LossMinimizationCallback",
                                "key": "loss",
                            },
                            "test_logits_type": {"_target_": "TensorTypeChecker", "key": "logits"},
                        },
                    },
                },
            }
        )
        runner.get_datasets = lambda *args, **kwargs: {
            "train": dataset,
            "valid": dataset,
        }
        runner.run()


@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required),
    reason="CUDA device is not available or no AMP found",
)
def test_experiment_engine_with_devices():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        train_from_runner(device)


@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required),
    reason="CUDA device is not available or no AMP found",
)
def test_config_experiment_engine_with_cuda():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        train_from_config(device)
