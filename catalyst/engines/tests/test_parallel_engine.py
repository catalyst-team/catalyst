# flake8: noqa

from typing import Any, Dict, List
import logging
import shutil

from pytest import mark
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from catalyst import dl

# from catalyst.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder

# from catalyst.dl import SupervisedRunner
from catalyst.core.runner import IRunner
from catalyst.engines import DataParallelEngine
from catalyst.registry import REGISTRY

# from catalyst.experiments import ConfigExperiment, Experiment
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .test_device_engine import (
    DummyDataset,
    DummyModel,
    LossMinimizationCallback,
    SupervisedRunner,
)

logger = logging.getLogger(__name__)


class CustomExperiment(dl.IExperiment):
    @property
    def seed(self) -> int:
        return 73

    @property
    def name(self) -> str:
        return "experiment73"

    @property
    def hparams(self) -> Dict:
        return {}

    @property
    def stages(self) -> List[str]:
        return ["train"]

    def get_stage_params(self, stage: str) -> Dict[str, Any]:
        return {
            "num_epochs": 10,
            "migrate_model_from_previous_stage": False,
            "migrate_callbacks_from_previous_stage": False,
        }

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, Any]:
        dataset = DummyDataset(10)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return DummyModel(4, 2)

    def get_criterion(self, stage: str):
        return nn.MSELoss()

    def get_optimizer(self, stage: str, model):
        return optim.SGD(model.parameters(), lr=1e-3)

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss"),
        }

    def get_engine(self):
        return DataParallelEngine()

    def get_trial(self):
        return None

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir="./logdir"),
        }


def run_train_with_experiment_parallel_device():
    # dataset = DummyDataset(10)
    # loader = DataLoader(dataset, batch_size=4)
    # runner = SupervisedRunner()
    # exp = Experiment(
    #     model=_model_fn,
    #     criterion=nn.MSELoss(),
    #     optimizer=_optimizer_fn,
    #     loaders={"train": loader, "valid": loader},
    #     main_metric="loss",
    #     callbacks=[
    #         CriterionCallback(),
    #         OptimizerCallback(),
    #         # DeviceCheckCallback(device),
    #         LossMinimizationCallback(),
    #     ],
    #     engine=DataParallelEngine(),
    # )
    runner = SupervisedRunner()
    experiment = CustomExperiment()
    runner.run(experiment)


def run_train_with_config_experiment_parallel_device():
    pass
    # dataset = DummyDataset(10)
    # runner = SupervisedRunner()
    # logdir = f"./test_dp_engine"
    # exp = ConfigExperiment(
    #     config={
    #         "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1,},
    #         "engine": "dp",
    #         "args": {"logdir": logdir},
    #         "stages": {
    #             "data_params": {"batch_size": 4, "num_workers": 0},
    #             "criterion_params": {"_target_": "MSELoss"},
    #             "optimizer_params": {"_target_": "SGD", "lr": 1e-3},
    #             "stage1": {
    #                 "stage_params": {"num_epochs": 2},
    #                 "callbacks_params": {
    #                     "loss": {"_target_": "CriterionCallback"},
    #                     "optimizer": {"_target_": "OptimizerCallback"},
    #                     # "test_device": {
    #                     #     "_target_": "DeviceCheckCallback",
    #                     #     "assert_device": str(device),
    #                     # },
    #                     "test_loss_minimization": {"callback": "LossMinimizationCallback"},
    #                 },
    #             },
    #         },
    #     }
    # )
    # exp.get_datasets = lambda *args, **kwargs: {
    #     "train": dataset,
    #     "valid": dataset,
    # }
    # runner.run(exp)
    # shutil.rmtree(logdir, ignore_errors=True)


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_experiment_parallel_engine_with_cuda():
    run_train_with_experiment_parallel_device()


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_config_experiment_engine_with_cuda():
    run_train_with_config_experiment_parallel_device()
