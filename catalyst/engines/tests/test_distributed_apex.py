# flake8: noqa

from typing import Any, Dict, List
import logging
import os
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

if SETTINGS.apex_required:
    from catalyst.engines.apex import DistributedDataParallelApexEngine

from .misc import (
    DummyDataset,
    DummyModel,
    LossMinimizationCallback,
    OPTTensorTypeChecker,
    WorldSizeCheckCallback,
)

logger = logging.getLogger(__name__)


if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


OPT_LEVELS = ("O0", "O1", "O2", "O3")


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, opt_level, port="12345"):
        super().__init__()
        self._logdir = logdir
        self._opt_level = opt_level
        self._port = port

    def get_engine(self) -> dl.IEngine:
        return DistributedDataParallelApexEngine(port=self._port, opt_level=self._opt_level)

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            # "checkpoint": dl.CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            # "check": DeviceCheckCallback(),
            "test_loss_minimization": LossMinimizationCallback("loss", logger=logger),
            "test_world_size": WorldSizeCheckCallback(NUM_CUDA_DEVICES, logger=logger),
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
        return {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_distributed_parallel_apex():
    for idx, opt_level in enumerate(OPT_LEVELS):
        with TemporaryDirectory() as logdir:
            runner = CustomRunner(logdir, opt_level, str(22222 + idx))
            runner.run()


class MyConfigRunner(dl.SupervisedConfigRunner):
    _dataset = DummyDataset(6)

    def get_datasets(self, *args, **kwargs):
        return {"train": self._dataset, "valid": self._dataset}


def _train_ddp_apex(port, logdir, opt_lvl):
    opt = str(opt_lvl).strip().upper()
    runner = MyConfigRunner(
        config={
            "args": {"logdir": logdir},
            "model": {"_target_": "DummyModel", "in_features": 4, "out_features": 2},
            "engine": {
                "_target_": "DistributedDataParallelApexEngine",
                "port": port,
                "opt_level": opt,
            },
            "args": {"logdir": logdir},
            "loggers": {"console": {"_target_": "ConsoleLogger"}},
            "stages": {
                "stage1": {
                    "num_epochs": 10,
                    "loaders": {"batch_size": 4, "num_workers": 0},
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
                        # "test_device": {"_target_": "DeviceCheckCallback", "assert_device": device},
                        "test_loss_minimization": {
                            "_target_": "LossMinimizationCallback",
                            "key": "loss",
                        },
                        "test_world_size": {
                            "_target_": "WorldSizeCheckCallback",
                            "assert_world_size": NUM_CUDA_DEVICES,
                        },
                        "test_logits_type": {
                            "_target_": "OPTTensorTypeChecker",
                            "key": "logits",
                            "opt_level": opt,
                        },
                    },
                },
            },
        }
    )
    runner.run()


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_config_train_distributed_parallel_apex():
    for idx, opt_level in enumerate(OPT_LEVELS):
        with TemporaryDirectory() as logdir:
            _train_ddp_apex(str(33333 + idx), logdir, opt_level)
