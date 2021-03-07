# flake8: noqa

from typing import Any, Dict, List
import logging
import os
import random
from tempfile import TemporaryDirectory

from pytest import mark
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from catalyst.callbacks import (
    AccuracyCallback,
    AUCCallback,
    CheckpointCallback,
    CriterionCallback,
    OptimizerCallback,
)
from catalyst.core.runner import IRunner
from catalyst.engines.device import DeviceEngine
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.engines.tests.misc import AllwaysSameModel, TwoBlobsDataset
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

logger = logging.getLogger(__name__)


if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


_BATCH_SIZE = 4
_WORKERS = 1
_LR = 1e-3


# experiment definition
class CustomDeviceRunner(IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return DeviceEngine(self._device)

    def get_loggers(self):
        return {"console": ConsoleLogger(), "csv": CSVLogger(logdir=self._logdir)}

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 1

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        dataset = TwoBlobsDataset()
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS, shuffle=False)
        return {"valid": loader}

    def get_model(self, stage: str):
        return AllwaysSameModel()

    def get_criterion(self, stage: str):
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, model, stage: str):
        return torch.optim.SGD(model.parameters(), lr=_LR)

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_callbacks(self, stage: str):
        return {
            "criterion": CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "accuracy": AccuracyCallback(input_key="logits", target_key="targets", num_classes=1),
            "auc": AUCCallback(input_key="scores", target_key="targets_onehot"),
            "optimizer": OptimizerCallback(metric_key="loss"),
            "checkpoint": CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y.view(-1),
            "targets_onehot": F.one_hot(y.view(-1), 2).to(torch.float32),
            "logits": logits,
            "scores": torch.sigmoid(logits),
        }


class CustomDDPRunner(IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_engine(self):
        return DistributedDataParallelEngine(port="22222")

    def get_loggers(self):
        return {"console": ConsoleLogger(), "csv": CSVLogger(logdir=self._logdir)}

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 1

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, Any]:
        dataset = TwoBlobsDataset()
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS, sampler=sampler)
        return {"valid": loader}

    def get_model(self, stage: str):
        return AllwaysSameModel()

    def get_criterion(self, stage: str):
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, model, stage: str):
        return torch.optim.SGD(model.parameters(), lr=_LR)

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_callbacks(self, stage: str):
        return {
            "criterion": CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "accuracy": AccuracyCallback(input_key="logits", target_key="targets", num_classes=1),
            "auc": AUCCallback(input_key="scores", target_key="targets_onehot"),
            "optimizer": OptimizerCallback(metric_key="loss"),
            "checkpoint": CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y.view(-1),
            "targets_onehot": F.one_hot(y.view(-1), 2).to(torch.float32),
            "logits": logits,
            "scores": torch.sigmoid(logits),
        }


class MyConfigRunner(SupervisedConfigRunner):
    _dataset = TwoBlobsDataset()

    def get_datasets(self, *args, **kwargs):
        return {"valid": self._dataset}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y.view(-1), "logits": logits}


def train_device_custom_runner(logdir, device):
    runner = CustomDeviceRunner(logdir, device)
    runner.run()
    return runner.epoch_metrics


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
                    "num_epochs": 1,
                    "loaders": {
                        "batch_size": _BATCH_SIZE * NUM_CUDA_DEVICES,
                        "num_workers": _WORKERS,
                    },
                    "criterion": {"_target_": "CrossEntropyLoss"},
                    "optimizer": {"_target_": "SGD", "lr": _LR},
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
    return runner.epoch_metrics


def train_ddp_custom_runner(logdir):
    runner = CustomDDPRunner(logdir)
    runner.run()
    return runner.epoch_metrics


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES == 2), reason="Number of CUDA devices is not 2",
)
def test_device_and_ddp_metrics():
    # we have to keep dataset_len, num_gpu and batch size synced
    # dataset is 64 points
    # in DP setup we have 64 / bs {4} = 16 iterations
    # in DDP setup we have 64 / (bs {4} * num_gpu {2}) = 8 iterations
    with TemporaryDirectory() as logdir:
        # logdir = "metrics_logs"
        device_logdir = os.path.join(logdir, "device_logs")
        ddp_logdir = os.path.join(logdir, "ddp_logs")

        device = "cuda:0"
        epoch_metrics1 = train_device_custom_runner(device_logdir, device)
        epoch_metrics2 = train_ddp_custom_runner(ddp_logdir)

        # print(f"epoch_metrics1: {epoch_metrics1}")
        # print(f"epoch_metrics2: {epoch_metrics2}")
        # assert 0 == 1
