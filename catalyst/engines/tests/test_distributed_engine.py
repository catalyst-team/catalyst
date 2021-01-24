# flake8: noqa

from typing import Any, Dict, List
import logging
import os
import shutil

from pytest import mark
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from catalyst import dl
from catalyst.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder

# from catalyst.dl import SupervisedRunner
from catalyst.engines import DistributedDataParallelEngine
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

if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


@REGISTRY.add
class WorldSizeCheckCallback(Callback):
    def __init__(self, assert_world_size: int):
        super().__init__(order=CallbackOrder.internal)
        self.world_size = assert_world_size

    def on_batch_start(self, runner: "IRunner"):
        device = runner.engine.device
        world_size = runner.engine.world_size
        logger.warning(
            f"WorldSizeCheckCallback: expected world size ({self.world_size}) - actual ({world_size})"
        )
        assert device < self.world_size
        assert self.world_size == world_size


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
        return DummyModel(4, 1)

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
            "check_world_size": WorldSizeCheckCallback(NUM_CUDA_DEVICES),
        }

    def get_engine(self):
        return DistributedDataParallelEngine()

    def get_trial(self):
        return None

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir="./logdir"),
        }


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_experiment_distributed_parallel_device():
    # logdir = "./test_ddp_engine"
    # dataset = DummyDataset(10)
    # # sampler = DistributedSampler(dataset, world_size, rank)
    # loader = DataLoader(dataset, batch_size=4)  # , sampler=sampler)
    # runner = SupervisedRunner()
    # engine = DistributedDataParallelEngine()
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
    #     logdir=logdir,
    #     engine=engine,
    # )
    runner = SupervisedRunner(engine=DistributedDataParallelEngine())
    experiment = CustomExperiment()
    runner.run(experiment)
    # shutil.rmtree(logdir, ignore_errors=True)


def _get_loaders(*args, **kwargs):
    dataset = DummyDataset(10)
    # sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, batch_size=4)  # , sampler=sampler)
    return {"train": loader, "valid": loader}


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_config_experiment_distributed_parallel_device():
    pass
    # logdir = "./test_config_ddp_engine"
    # runner = SupervisedRunner()
    # exp = ConfigExperiment(
    #     config={
    #         "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1,},
    #         "engine": "ddp",
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
    #                     "test_loss_minimization": {"_target_": "LossMinimizationCallback"},
    #                 },
    #             },
    #         },
    #     }
    # )
    # exp.get_loaders = _get_loaders
    # # CORE
    # runner.run_experiment(exp)
    # shutil.rmtree(logdir, ignore_errors=True)
