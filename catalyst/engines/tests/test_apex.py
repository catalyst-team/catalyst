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
    OPTTensorTypeChecker,
)

if SETTINGS.apex_required:
    from catalyst.engines.apex import APEXEngine

logger = logging.getLogger(__name__)


OPT_LEVELS = (
    "O0",
    # "O1",  # disabled, issue: https://github.com/NVIDIA/apex/issues/694
    "O2",
    "O3",
)


class CustomRunner(IRunner):
    def __init__(self, logdir, device, opt_level):
        super().__init__()
        self._logdir = logdir
        self._device = device
        self._opt_level = opt_level

    def get_engine(self):
        return APEXEngine(self._device, apex_kwargs=dict(opt_level=self._opt_level))

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
            "test_logits_type": OPTTensorTypeChecker("logits", self._opt_level),
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


def train_from_runner(device, opt_level):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device, opt_level)
        runner.run()


def train_from_config(device, opt_level):
    with TemporaryDirectory() as logdir:
        dataset = DummyDataset(6)
        runner = SupervisedConfigRunner(
            config={
                "args": {"logdir": logdir},
                "model": {"_target_": "DummyModel", "in_features": 4, "out_features": 2},
                "engine": {
                    "_target_": "APEXEngine",
                    "device": device,
                    "apex_kwargs": {"opt_level": opt_level.upper()},
                },
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
                            "test_opt_logits_type": {
                                "_target_": "OPTTensorTypeChecker",
                                "key": "logits",
                                "opt_level": opt_level,
                            },
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
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required),
    reason="CUDA device is not available or no Apex found",
)
def test_apex_with_devices():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        for level in OPT_LEVELS:
            train_from_runner(device, level)


@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required),
    reason="CUDA device is not available or no Apex found",
)
def test_config_apex_with_devices():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        for level in OPT_LEVELS:
            train_from_config(device, level)
