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

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """
    Dummy dataset.
    """

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
    def __init__(self, assert_device: str):
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)

    def on_stage_start(self, runner: "IRunner"):
        model_device = next(runner.model.parameters()).device
        logger.warning(
            f"DeviceCheckCallback: model device ({model_device}) - device ({self.device})"
        )
        assert model_device == self.device


@REGISTRY.add
class LossMinimizationCallback(Callback):
    def __init__(self, key="loss"):
        super().__init__(order=CallbackOrder.metric)
        self.key = key
        self.container = None
        self.round_nums = 6

    def on_epoch_start(self, runner: "IRunner"):
        self.container = []

    def on_batch_end(self, runner: "IRunner"):
        self.container.append(runner.batch_metrics[self.key].item())

    def on_epoch_end(self, runner: "IRunner"):
        assert len(self.container)

        container = [round(num, self.round_nums) for num in self.container]
        logger.warning(f"LossMinimizationCallback: {container}")
        assert all(round(a, 6) >= round(b, 6) for a, b in zip(container[:-1], container[1:]))

        self.container = []
