# flake8: noqa
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import REGISTRY


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
    """Docs."""

    def __init__(self, in_features, out_features):
        """Docs."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Linear(in_features, out_features)

    def forward(self, batch):
        """Docs."""
        return self.layers(batch)


class TwoBlobsDataset(Dataset):
    def __init__(self):
        """
        Dataset was generated with sklearn help:

        >>> from sklearn.datasets import make_blobs
        >>> x, y = make_blobs(n_samples=50, centers=2, n_features=4, random_state=0)
        >>> print(x)
        >>> print(y)

        """
        self.points = [
            [1.0427873, 4.60625923, 1.42094543, 0.53492249],
            [0.06897171, 4.35573272, 2.78435808, 1.02664657],
            [-2.84281142, 2.45629766, -1.31649738, 9.54880274],
            [-0.88677249, 1.30092622, -1.2725819, 7.09742911],
            [-0.9503132, 2.70958351, -0.85224906, 6.74239851],
            [2.50904929, 5.7731461, 2.21021495, 1.27582618],
            [-0.90167256, 1.31582461, -2.35263911, 7.88762509],
            [-0.07228289, 2.88376939, 0.34899733, 2.84843906],
            [-2.3881297, 4.82794721, -1.51625915, 8.63791641],
            [0.30380963, 3.94423417, 1.24212124, -0.82861894],
            [0.46546494, 3.12315514, 2.02708529, 1.32599553],
            [-0.18887976, 5.20461381, 2.52092996, -0.63858003],
            [-0.40026809, 1.83795075, -2.39572443, 7.39763997],
            [-0.63762777, 4.09104705, 1.15980096, 1.28456616],
            [1.15369622, 3.90200639, 0.42506917, 1.36044592],
            [-1.57671974, 4.95740592, 2.91970372, 0.15549864],
            [-1.88089792, 1.54293097, -1.89187418, 5.61205686],
            [-1.15047848, 1.81848147, -0.9500176, 9.16184591],
            [-0.33887422, 3.23482487, -0.32739695, 8.15418767],
            [0.87305123, 4.71438583, 2.19931109, 2.35193717],
            [-0.6700734, 2.26685667, -2.28249862, 8.51705453],
            [-1.89608585, 2.67850308, -0.14859618, 8.49072375],
            [0.4666179, 3.86571303, 0.80247216, 1.67515402],
            [-0.75511346, 3.74138642, 0.91498017, 9.17198797],
            [-2.75233953, 3.76224524, -2.24847112, 6.29068892],
            [0.9867701, 6.08965782, 2.18217961, 1.29965302],
            [-2.20123667, 2.94971282, -1.88410185, 8.51189331],
            [0.85624076, 3.86236175, -2.161078, 8.9524763],
            [1.18454506, 5.28042636, 2.41163392, 1.60423683],
            [2.46452227, 6.1996765, 3.23404709, 0.71773882],
            [1.7373078, 4.42546234, 2.49913075, 1.23133799],
            [-2.22147187, 2.76824772, -1.68340933, 9.68472374],
            [-0.57965205, 2.76287217, -0.6341764, 8.75766669],
            [-2.02493646, 4.84741432, -0.29883497, 7.92301126],
            [-3.01816161, 3.35727396, -1.08158228, 8.47049145],
            [-1.56618683, 1.74978876, -0.72497911, 7.66391368],
            [-2.26646701, 4.46089686, -2.54111268, 8.10251088],
            [-0.85460926, 3.3253441, -2.01817185, 8.37470921],
            [-2.33031368, 2.22833248, -1.70378828, 7.85293917],
            [-2.27165884, 2.09144372, -1.3467083, 7.17198173],
            [3.2460247, 2.84942165, 2.10102604, 0.71047981],
            [-0.19685333, 6.24740851, 1.64164854, 0.15020885],
            [2.85942078, 2.95602827, 0.78478252, 1.86706037],
            [2.11567076, 3.06896151, 2.45760916, 0.21285357],
            [2.47034915, 4.09862906, 2.36833522, 0.04356792],
            [0.39603801, 4.39839705, 0.61930319, 8.74150467],
            [-0.09448254, 5.35823905, 1.65209057, 2.12010873],
            [2.20656076, 5.50616718, 1.6679407, 0.59536091],
            [0.10547293, 3.72493766, 1.74371499, 0.953829],
            [0.08848433, 2.32299086, 1.70735537, 1.05401263],
        ]
        # fmt: off
        self.labels = [
            0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        ]
        # fmt: on
        assert len(self.points) == len(self.labels)

    def __len__(self):
        """
        Returns:
            dataset's length.
        """
        return len(self.points)

    def __getitem__(self, index):
        """
        Args:
            idx: index of sample

        Returns:
            dummy features and targets vector
        """
        x = torch.FloatTensor(self.points[index])
        y = torch.LongTensor([self.labels[index]])
        return x, y


@REGISTRY.add
class AllwaysSameModel(nn.Module):
    """Docs"""

    def __init__(self, *args, **kwargs):
        """Docs."""
        super().__init__()
        self.in_features = 4
        self.out_features = 2
        self.layers = nn.Linear(self.in_features, self.out_features)
        # set initial layers weights
        weights = torch.nn.Parameter(
            torch.FloatTensor(
                [[0.3429, 0.0886, 0.4281, -0.0090], [-0.1003, -0.2247, -0.2584, -0.1986]],
            )
        )
        weights.requires_grad = True
        self.layers.weight = weights

    def forward(self, batch):
        """Docs"""
        return self.layers(batch)


@REGISTRY.add
class DeviceCheckCallback(Callback):
    """Docs."""

    def __init__(self, assert_device: str, logger=None):
        """Docs."""
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_stage_start(self, runner: "IRunner"):
        """Docs."""
        model_device = next(runner.model.parameters()).device
        self.logger.warning(
            f"DeviceCheckCallback: model device ({model_device}) - device ({self.device})"
        )
        assert model_device == self.device


@REGISTRY.add
class LossMinimizationCallback(Callback):
    """Docs."""

    def __init__(self, key="loss", nums=5, logger=None):
        """Docs."""
        super().__init__(order=CallbackOrder.metric)
        self.key = key
        self.container = None
        self.nums = nums
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_epoch_start(self, runner: "IRunner"):
        """Docs."""
        self.container = []

    def on_batch_end(self, runner: "IRunner"):
        """Docs."""
        self.container.append(runner.batch_metrics[self.key].item())

    def on_epoch_end(self, runner: "IRunner"):
        """Docs."""
        assert len(self.container)

        container = [round(num, self.nums) for num in self.container]
        self.logger.warning(f"LossMinimizationCallback: {container}")
        assert all(
            round(a, self.nums) >= round(b, self.nums)
            for a, b in zip(container[:-1], container[1:])
        )

        self.container = []


@REGISTRY.add
class WorldSizeCheckCallback(Callback):
    """Docs."""

    def __init__(self, assert_world_size: int, logger=None):
        """Docs."""
        super().__init__(order=CallbackOrder.internal)
        self.world_size = assert_world_size
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def on_batch_start(self, runner: "IRunner"):
        """Docs."""
        rank = runner.engine.rank
        world_size = runner.engine.world_size
        self.logger.warning(
            f"WorldSizeCheckCallback: "
            f"expected world size ({self.world_size}) - actual ({world_size})"
        )
        assert rank < self.world_size
        assert self.world_size == world_size


@REGISTRY.add
class TensorTypeChecker(Callback):
    """Docs."""

    def __init__(self, key, use_batch_metrics=False):
        """Docs."""
        super().__init__(CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics

    def on_batch_end(self, runner):
        """Docs."""
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
    """Docs."""

    def __init__(self, key, opt_level, use_batch_metrics=False):
        """Docs."""
        super().__init__(CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics
        self.opt_level = opt_level
        self.expected_type = OPT_TYPE_MAP[opt_level]

    def on_batch_end(self, runner):
        """Docs."""
        check_tensor = (
            runner.batch_metrics[self.key] if self.use_batch_metrics else runner.batch[self.key]
        )
        assert check_tensor.dtype == self.expected_type, (
            f"Wrong types for {self.opt_level} - actual is "
            f"'{check_tensor.dtype}' but expected is '{self.expected_type}'!"
        )
