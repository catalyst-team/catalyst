# flake8: noqa

from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst.callbacks import CheckpointCallback, CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.engines.apex import APEXEngine
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .misc import (
    DeviceCheckCallback,
    DummyDataset,
    DummyModel,
    LossMinimizationCallback,
    OPTTensorTypeChecker,
)

logger = logging.getLogger(__name__)


OPT_LEVELS = ("O0", "O1", "O2", "O3")


class CustomRunner(IRunner):
    def __init__(self, logdir, device, opt_level):
        super().__init__()
        self._logdir = logdir
        self._device = device
        self._opt_level = opt_level

    def get_engine(self):
        return APEXEngine(self._device, self._opt_level)

    def get_callbacks(self, stage: str) -> Dict[str, Callback]:
        return {
            "criterion": CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            # TODO: fix issue with pickling wrapped model's forward function
            # "checkpoint": dl.CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            "check": DeviceCheckCallback(self._device, logger=logger),
            "check2": LossMinimizationCallback("loss", logger=logger),
            "logits_type_checker": OPTTensorTypeChecker("logits", self._opt_level),
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


def run_train_with_experiment_apex_device(device, opt_level):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device, opt_level)
        runner.run()


def run_train_with_config_experiment_apex_device(device, opt_level):
    pass


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA devices is not available")
def test_apex_with_devices():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        for level in OPT_LEVELS:
            run_train_with_experiment_apex_device(device, level)


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA devices is not available")
def test_config_apex_with_devices():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        for level in OPT_LEVELS:
            run_train_with_config_experiment_apex_device(device, level)
