# flake8: noqa


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

from catalyst.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.dl import SupervisedRunner
from catalyst.engines import DistributedDataParallelEngine
from catalyst.experiments import ConfigExperiment, Experiment
from catalyst.registry import REGISTRY
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


class DummyDataset(Dataset):
    """
    Dummy dataset.
    """

    features_dim: int = 4
    out_dim: int = 1

    def __init__(self, num_records: int):
        self.num_records = num_records

    def __len__(self):
        """
        Returns:
            dataset's length.
        """
        return self.num_records

    def __getitem__(self, idx: int):
        """
        Args:
            idx: index of sample

        Returns:
            dummy features and targets vector
        """
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


@REGISTRY.add
class DummyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Linear(in_features, out_features)

    def forward(self, batch):
        return self.layers(batch)


def _model_fn():
    return DummyModel(4, 1)


def _optimizer_fn(parameters):
    return optim.SGD(parameters, lr=1e-3)


@REGISTRY.add
class DeviceCheckCallback(Callback):
    def __init__(self, assert_device: str):
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)

    def on_stage_start(self, runner: "IRunner"):
        model_device = next(runner.model.parameters()).device
        assert model_device == self.device


@REGISTRY.add
class LossMinimizationCallback(Callback):
    def __init__(self, key="loss"):
        super().__init__(order=CallbackOrder.metric)
        self.key = key
        self.container = None

    def on_epoch_start(self, runner: "IRunner"):
        self.container = []

    def on_batch_end(self, runner: "IRunner"):
        self.container.append(runner.batch_metrics[self.key].item())

    def on_epoch_end(self, runner: "IRunner"):
        print(self.container)
        assert all(a >= b for a, b in zip(self.container[:-1], self.container[1:]))
        self.container = []


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_experiment_distributed_parallel_device():
    logdir = "./test_ddp_engine"
    dataset = DummyDataset(10)
    # sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, batch_size=4)  # , sampler=sampler)
    runner = SupervisedRunner()
    engine = DistributedDataParallelEngine()
    exp = Experiment(
        model=_model_fn,
        criterion=nn.MSELoss(),
        optimizer=_optimizer_fn,
        loaders={"train": loader, "valid": loader},
        main_metric="loss",
        callbacks=[
            CriterionCallback(),
            OptimizerCallback(),
            # DeviceCheckCallback(device),
            LossMinimizationCallback(),
        ],
        logdir=logdir,
        engine=engine,
    )
    runner.run_experiment(exp)
    shutil.rmtree(logdir, ignore_errors=True)


def _get_loaders(*args, **kwargs):
    dataset = DummyDataset(10)
    # sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, batch_size=4)  # , sampler=sampler)
    return {"train": loader, "valid": loader}


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_config_experiment_distributed_parallel_device():
    logdir = "./test_config_ddp_engine"
    runner = SupervisedRunner()
    exp = ConfigExperiment(
        config={
            "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1,},
            "engine": "ddp",
            "args": {"logdir": logdir},
            "stages": {
                "data_params": {"batch_size": 4, "num_workers": 0},
                "criterion_params": {"_target_": "MSELoss"},
                "optimizer_params": {"_target_": "SGD", "lr": 1e-3},
                "stage1": {
                    "stage_params": {"num_epochs": 2},
                    "callbacks_params": {
                        "loss": {"_target_": "CriterionCallback"},
                        "optimizer": {"_target_": "OptimizerCallback"},
                        # "test_device": {
                        #     "_target_": "DeviceCheckCallback",
                        #     "assert_device": str(device),
                        # },
                        "test_loss_minimization": {"_target_": "LossMinimizationCallback"},
                    },
                },
            },
        }
    )
    exp.get_loaders = _get_loaders
    # CORE
    runner.run_experiment(exp)
    shutil.rmtree(logdir, ignore_errors=True)
