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
from catalyst.runners.config import SupervisedConfigRunner
from catalyst.engines import DataParallelEngine
from catalyst.engines.device import DeviceEngine
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .misc import DummyDataset, DummyModel, LossMinimizationCallback

logger = logging.getLogger(__name__)


class CustomRunner(IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_engine(self):
        return DataParallelEngine()

    def get_callbacks(self, stage: str) -> Dict[str, Callback]:
        return {
            "criterion": CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss", logger=logger),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 10

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

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }

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


def run_train_with_experiment_parallel_device():
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir)
        runner.run()


def run_train_with_config_experiment_parallel_device():
    device = "dp"
    with TemporaryDirectory() as logdir:
        dataset = DummyDataset(6)
        runner = SupervisedConfigRunner(
            config={
                "args": {"logdir": logdir},
                "model": {"_target_": "DummyModel", "in_features": 4, "out_features": 2},
                "engine": {"engine": device},
                "args": {"logdir": logdir},
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
                            "test_loss_minimization": {"_target_": "LossMinimizationCallback", "key": "loss"},
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


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_experiment_parallel_engine_with_cuda():
    run_train_with_experiment_parallel_device()


# @mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_config_experiment_engine_with_cuda():
    run_train_with_config_experiment_parallel_device()
