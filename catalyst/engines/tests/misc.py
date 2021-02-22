from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks import CheckpointCallback, CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.engines.device import DeviceEngine
from catalyst.loggers import ConsoleLogger, CSVLogger
from catalyst.registry import REGISTRY
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


class DummyDataset(Dataset):
    """Dummy dataset."""

    features_dim: int = 4
    out_dim: int = 2

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


@REGISTRY.add
class DeviceCheckCallback(Callback):
    def __init__(self, assert_device: str, logger=None):
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_stage_start(self, runner: "IRunner"):
        model_device = next(runner.model.parameters()).device
        self.logger.warning(f"DeviceCheckCallback: model device ({model_device}) - device ({self.device})")
        assert model_device == self.device


@REGISTRY.add
class LossMinimizationCallback(Callback):
    def __init__(self, key="loss", nums=5, logger=None):
        super().__init__(order=CallbackOrder.metric)
        self.key = key
        self.container = None
        self.nums = nums
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_epoch_start(self, runner: "IRunner"):
        self.container = []

    def on_batch_end(self, runner: "IRunner"):
        self.container.append(runner.batch_metrics[self.key].item())

    def on_epoch_end(self, runner: "IRunner"):
        assert len(self.container)

        container = [round(num, self.nums) for num in self.container]
        self.logger.warning(f"LossMinimizationCallback: {container}")
        assert all(round(a, self.nums) >= round(b, self.nums) for a, b in zip(container[:-1], container[1:]))

        self.container = []


@REGISTRY.add
class WorldSizeCheckCallback(Callback):
    def __init__(self, assert_world_size: int, logger=None):
        super().__init__(order=CallbackOrder.internal)
        self.world_size = assert_world_size
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_batch_start(self, runner: "IRunner"):
        rank = runner.engine.rank
        world_size = runner.engine.world_size
        self.logger.warning(
            f"WorldSizeCheckCallback: " f"expected world size ({self.world_size}) - actual ({world_size})"
        )
        assert rank < self.world_size
        assert self.world_size == world_size


@REGISTRY.add
class TensorTypeChecker(Callback):
    def __init__(self, key, use_batch_metrics=False):
        super().__init__(CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics

    def on_batch_end(self, runner):
        if self.use_batch_metrics:
            assert runner.batch_metrics[self.key].dtype == torch.float16
        else:
            assert runner.batch[self.key].dtype == torch.float16


OPT_TYPE_MAP = {
    "O0": torch.float32,  # no-op training
    "O1": torch.float16,  # mixed precision (FP16) training
    "O2": torch.float32,  # almost FP16 training
    "O3": torch.float32,  # another implementation of FP16 training
}


@REGISTRY.add
class OPTTensorTypeChecker(Callback):
    def __init__(self, key, opt_level, use_batch_metrics=False):
        super().__init__(CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics
        self.opt_level = opt_level
        self.expected_type = OPT_TYPE_MAP[opt_level]

    def on_batch_end(self, runner):
        check_tensor = runner.batch_metrics[self.key] if self.use_batch_metrics else runner.batch[self.key]
        assert check_tensor.dtype == self.expected_type, (
            f"Wrong types for {self.opt_level} - actual is "
            f"'{check_tensor.dtype}' but expected is '{self.expected_type}'!"
        )
