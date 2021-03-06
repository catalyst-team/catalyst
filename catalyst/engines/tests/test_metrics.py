# flake8: noqa

from typing import Any, Dict, List
import logging
import os
import random
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst.callbacks import CheckpointCallback, CriterionCallback, OptimizerCallback
from catalyst.core.runner import IRunner
from catalyst.engines.device import DeviceEngine
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .misc import AllwaysSameModel, TwoBlobsDataset

logger = logging.getLogger(__name__)


if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


_BATCH_SIZE = 2
_WORKERS = 1


# experiment definition
class CustomDeviceRunner(IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return DeviceEngine(self._device)

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
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 10

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        dataset = TwoBlobsDataset()
        loader = DataLoader(
            dataset, batch_size=_BATCH_SIZE * NUM_CUDA_DEVICES, num_workers=_WORKERS
        )
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return AllwaysSameModel()

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


class CustomDDPRunner(IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_engine(self):
        return DistributedDataParallelEngine(port="22222")

    def get_callbacks(self, stage: str):
        return {
            "criterion": CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": OptimizerCallback(metric_key="loss"),
            "checkpoint": CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        }

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 10

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, Any]:
        dataset = TwoBlobsDataset()
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return AllwaysSameModel()

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


class MyConfigRunner(SupervisedConfigRunner):
    _dataset = TwoBlobsDataset()

    def get_datasets(self, *args, **kwargs):
        return {"train": self._dataset, "valid": self._dataset}


def train_device_custom_runner(logdir, device):
    runner = CustomDeviceRunner(logdir, device)
    runner.run()


def train_device_config_runner(logdir, device):
    runner = MyConfigRunner(
        config={
            "args": {"logdir": logdir},
            "model": {"_target_": "AllwaysSameModel"},
            "engine": {"_target_": "DeviceEngine", "device": device},
            "args": {"logdir": logdir},
            "loggers": {"console": {"_target_": "ConsoleLogger"}},
            "stages": {
                "stage1": {
                    "num_epochs": 10,
                    "loaders": {
                        "batch_size": _BATCH_SIZE * NUM_CUDA_DEVICES,
                        "num_workers": _WORKERS,
                    },
                    "criterion": {"_target_": "MSELoss"},
                    "optimizer": {"_target_": "Adam", "lr": 1e-3},
                    "callbacks": {
                        "criterion": {
                            "_target_": "CriterionCallback",
                            "metric_key": "loss",
                            "input_key": "logits",
                            "target_key": "targets",
                        },
                        "optimizer": {"_target_": "OptimizerCallback", "metric_key": "loss"},
                    },
                },
            },
        }
    )
    runner.run()


def train_ddp_custom_runner(logdir):
    runner = CustomDDPRunner(logdir)
    runner.run()


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_device_and_ddp_metrics():
    # with TemporaryDirectory() as logdir:
    logdir = "metrics_logs"

    device_logdir = os.path.join(logdir, "device_logs")
    ddp_logdir = os.path.join(logdir, "ddp_logs")

    device = "cuda:0"
    train_device_config_runner(device_logdir, device)

    train_ddp_custom_runner(ddp_logdir)
