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
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

if SETTINGS.apex_required:
    from catalyst.engines.apex import DistributedDataParallelApexEngine

from .misc import (  # DistributedDataParallelTypeChecker,
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
DDP_ADDRESS = random.randint(22222, 99999)


class CustomRunner(IRunner):
    def __init__(self, logdir, opt_level, port="12345"):
        super().__init__()
        self._logdir = logdir
        self._opt_level = opt_level
        self._port = port

    def get_engine(self):
        return DistributedDataParallelApexEngine(
            port=DDP_ADDRESS + random.randint(1, 100),
            process_group_kwargs={"backend": "nccl"},
            apex_kwargs=dict(opt_level=self._opt_level),
        )

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
            # "test_nn_parallel_distributed_data_parallel": DistributedDataParallelTypeChecker(),
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
        return {"console": ConsoleLogger(), "csv": CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


class MyConfigRunner(SupervisedConfigRunner):
    _dataset = DummyDataset(6)

    def get_datasets(self, *args, **kwargs):
        return {"train": self._dataset, "valid": self._dataset}


def train_from_config(port, logdir, opt_lvl):
    opt = str(opt_lvl).strip().upper()
    runner = MyConfigRunner(
        config={
            "args": {"logdir": logdir},
            "model": {"_target_": "DummyModel", "in_features": 4, "out_features": 2},
            "engine": {
                "_target_": "DistributedDataParallelApexEngine",
                "port": DDP_ADDRESS + random.randint(100, 200),
                "process_group_kwargs": {"backend": "nccl"},
                "apex_kwargs": {"opt_level": opt},
            },
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
                        # "test_nn_parallel_distributed_data_parallel": {
                        #     "_target_": "DistributedDataParallelTypeChecker"
                        # },
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
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="Number of CUDA devices is less than 2 or no Apex found",
)
def test_train_distributed_parallel_apex():
    for idx, opt_level in enumerate(OPT_LEVELS):
        with TemporaryDirectory() as logdir:
            runner = CustomRunner(logdir, opt_level, DDP_ADDRESS + random.randint(1, 100))
            runner.run()


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="Number of CUDA devices is less than 2 or no Apex found",
)
def test_config_train_distributed_parallel_apex():
    for idx, opt_level in enumerate(OPT_LEVELS):
        with TemporaryDirectory() as logdir:
            train_from_config(DDP_ADDRESS + random.randint(100, 200), logdir, opt_level)
