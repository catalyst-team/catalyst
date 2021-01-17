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

from catalyst import registry
from catalyst.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.dl import SupervisedRunner
from catalyst.engines import DistributedDataParallelEngine
from catalyst.experiments import ConfigExperiment, Experiment
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


@registry.Model
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


@registry.Callback
class DeviceCheckCallback(Callback):
    def __init__(self, assert_device: str):
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)

    def on_stage_start(self, runner: "IRunner"):
        model_device = next(runner.model.parameters()).device
        assert model_device == self.device


@registry.Callback
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
        assert all(
            a >= b for a, b in zip(self.container[:-1], self.container[1:])
        )
        self.container = []


def run_train_with_experiment_distributed_parallel_device(rank, world_size):
    logdir = "./test_ddp_engine"
    dataset = DummyDataset(10)
    sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    runner = SupervisedRunner(device=rank)
    engine = DistributedDataParallelEngine(rank, world_size)
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
    # CORE
    engine.setup_experiment()
    runner.run(exp)
    engine.cleanup()
    shutil.rmtree(logdir, ignore_errors=True)


def run_train_with_config_experiment_distributed_parallel_device(
    rank, world_size
):
    logdir = "./test_config_ddp_engine"
    dataset = DummyDataset(10)
    sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    runner = SupervisedRunner(device=rank)
    exp = ConfigExperiment(
        config={
            "model_params": {
                "model": "DummyModel",
                "in_features": 4,
                "out_features": 1,
            },
            "engine": "ddp",
            "args": {"logdir": logdir},
            "stages": {
                "data_params": {"batch_size": 4, "num_workers": 0},
                "criterion_params": {"criterion": "MSELoss"},
                "optimizer_params": {"optimizer": "SGD", "lr": 1e-3},
                "stage1": {
                    "stage_params": {"num_epochs": 2},
                    "callbacks_params": {
                        "loss": {"callback": "CriterionCallback"},
                        "optimizer": {"callback": "OptimizerCallback"},
                        # "test_device": {
                        #     "callback": "DeviceCheckCallback",
                        #     "assert_device": str(device),
                        # },
                        "test_loss_minimization": {
                            "callback": "LossMinimizationCallback"
                        },
                    },
                },
            },
        }
    )
    exp.get_loaders = lambda *args, **kwargs: {
        "train": loader,
        "valid": loader,
    }
    # CORE
    # engine.setup_experiment()
    runner.run(exp)
    # engine.cleanup()
    shutil.rmtree(logdir, ignore_errors=True)


def _run_test(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2,
    reason="Number of CUDA devices is less than 2",
)
def test_experiment_distributed_parallel_engine_with_cuda():
    _run_test(
        run_train_with_experiment_distributed_parallel_device,
        NUM_CUDA_DEVICES,
    )


@mark.skip(
    "Need to rewrite runner so all of the DDP initializations will be inside!"
)
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2,
    reason="Number of CUDA devices is less than 2",
)
def test_config_experiment_distributed_parallel_engine_with_cuda():
    _run_test(
        run_train_with_config_experiment_distributed_parallel_device,
        NUM_CUDA_DEVICES,
    )
