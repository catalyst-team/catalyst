# flake8: noqa

from typing import Any, Dict, List
import logging
import os
import random
from tempfile import TemporaryDirectory
import time

from pytest import mark
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler

from catalyst.callbacks import (
    AccuracyCallback,
    AUCCallback,
    CheckpointCallback,
    CriterionCallback,
    OptimizerCallback,
    PrecisionRecallF1SupportCallback,
    TqdmCallback,
)
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.models import MnistSimpleNet
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder, CallbackScope
from catalyst.core.runner import IRunner
from catalyst.data.transforms import ToTensor
from catalyst.engines.tests.misc import TwoBlobsDataset, TwoBlobsModel
from catalyst.engines.torch import DataParallelEngine, DeviceEngine, DistributedDataParallelEngine
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

logger = logging.getLogger(__name__)


if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


_BATCH_SIZE = 32
_WORKERS = 1
_LR = 1e-3


class CustomDistributedSampler(DistributedSampler):
    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


class CustomSampler(SequentialSampler):
    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[0 : len(self.data_source) : 2] + indices[1 : len(self.data_source) : 2]
        return iter(indices)


class CounterCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.external, CallbackNode.all, CallbackScope.stage)
        self.counter = 0
        self.counter2 = [0] * 4
        self.counter3 = 0

    def on_loader_start(self, runner: "IRunner") -> None:
        self.counter = 0
        self.counter2 = [0] * 4
        self.counter3 = 0

    def on_batch_end(self, runner: "IRunner") -> None:
        self.counter += torch.sum(runner.batch["targets"]).detach().cpu().item()
        preds = torch.argmax(runner.batch["logits"], dim=1).detach().cpu().numpy()
        for i_class in range(runner.batch["logits"].shape[1]):
            self.counter2[i_class] += (preds == i_class).sum()
        self.counter3 += len(runner.batch["logits"])

    def on_loader_end(self, runner: "IRunner") -> None:
        print(f"{runner.engine}, {self.counter}, {self.counter2}, {self.counter3}")
        # assert self.counter == self.required_num, f"{runner.engine}, {self.counter}"


class IRunnerMixin(IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_loggers(self):
        return {"console": ConsoleLogger(), "csv": CSVLogger(logdir=self._logdir)}

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 1

    def get_model(self, stage: str):
        # return MnistSimpleNet(out_features=10, normalize=False)
        return TwoBlobsModel()

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
            "accuracy": AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1,)),
            "auc": AUCCallback(input_key="scores", target_key="targets_onehot"),
            "classification": PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=4,
            ),
            # "optimizer": OptimizerCallback(metric_key="loss"),
            # "checkpoint": CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            # "verbose": TqdmCallback(),
            "counter": CounterCallback(),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)
        num_class = logits.shape[1]

        self.batch = {
            "features": x,
            "targets": y.view(-1),
            "targets_onehot": F.one_hot(y.view(-1), num_class).to(torch.float32),
            "logits": logits,
            "scores": torch.sigmoid(logits),
        }


class CustomDeviceRunner(IRunnerMixin, IRunner):
    def get_engine(self):
        return DeviceEngine("cuda:0")

    def get_loaders(self, stage: str):
        dataset = TwoBlobsDataset()
        # dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
        sampler = CustomSampler(data_source=dataset)
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS, sampler=sampler)
        return {"valid": loader}


class CustomDPRunner(IRunnerMixin, IRunner):
    def get_engine(self):
        return DataParallelEngine()

    def get_loaders(self, stage: str):
        dataset = TwoBlobsDataset()
        # dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
        sampler = CustomSampler(data_source=dataset)
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS, sampler=sampler)
        return {"valid": loader}


class CustomDDPRunner(IRunnerMixin, IRunner):
    def get_engine(self):
        return DistributedDataParallelEngine(port="22222")

    def get_loaders(self, stage: str):
        dataset = TwoBlobsDataset()
        # dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
        sampler = CustomDistributedSampler(dataset=dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=_BATCH_SIZE, num_workers=_WORKERS, sampler=sampler)
        return {"valid": loader}


class MyConfigRunner(SupervisedConfigRunner):
    _dataset = TwoBlobsDataset()

    def get_datasets(self, *args, **kwargs):
        return {"valid": self._dataset}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y.view(-1), "logits": logits}


def train_device_custom_runner(logdir):
    runner = CustomDeviceRunner(logdir)
    runner.run()
    return runner.epoch_metrics


def train_dp_custom_runner(logdir):
    runner = CustomDPRunner(logdir)
    runner.run()
    return runner.epoch_metrics


def train_ddp_custom_runner(logdir):
    runner = CustomDDPRunner(logdir)
    runner.run()
    return runner.epoch_metrics


def train_device_config_runner(logdir, device):
    runner = MyConfigRunner(
        config={
            "args": {"logdir": logdir},
            "model": {"_target_": "AllwaysSameModel"},
            "engine": {"_target_": "DeviceEngine", "device": device},
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


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES == 2), reason="Number of CUDA devices is not 2",
)
def test_device_and_ddp_metrics():
    # we have to keep dataset_len, num_gpu and batch size synced
    # dataset is 64 points
    # in DP setup we have 64 / (bs {32} * num_gpu {1} ) = 2 iteration on 1 gpu
    # in DDP setup we have 64 / (bs {32} * num_gpu {2}) = 1 iteration on 2 gpu
    with TemporaryDirectory() as logdir:
        # logdir = "metrics_logs"
        device_logdir = os.path.join(logdir, "device_logs")
        dp_logdir = os.path.join(logdir, "dp_logs")
        ddp_logdir = os.path.join(logdir, "ddp_logs")

        epoch_metrics1 = train_device_custom_runner(device_logdir)
        print("=" * 80)
        epoch_metrics2 = train_dp_custom_runner(dp_logdir)
        print("=" * 80)
        epoch_metrics3 = train_ddp_custom_runner(ddp_logdir)
        print("=" * 80)

        # print(f"epoch_metrics1: {epoch_metrics1}")
        # print(f"epoch_metrics2: {epoch_metrics2}")
        # assert 0 == 1
