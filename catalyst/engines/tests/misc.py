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
        >>> x, y = make_blobs(n_samples=64, centers=2, n_features=4, random_state=42)
        >>> x[2], x[-2], y[2], y[-2] = x[-2], x[2], y[-2], y[2]
        >>> print(x)
        >>> print(y)

        """
        # self.points = [
        #     [-5.12894273, 9.83618863, 4.7269259, 1.67416233],
        #     [-2.41212007, 9.98293112, 3.93782574, 1.64550754],
        #     [-1.59379551, 9.34303724, 4.11011863, 2.48643712],
        #     [-3.83738367, 9.21114736, 5.37834542, 2.14453797],
        #     [-3.6601912, 9.38998415, 4.03924015, 1.68147593],
        #     [-8.39899716, -7.36434367, -7.57141661, 6.61585345],
        #     [-3.6155326, 7.8180795, 5.45240466, 3.32940971],
        #     [-8.14051115, -5.96224765, -6.71617156, 8.35598818],
        #     [-7.60999382, -6.663651, -8.79275592, 6.67192257],
        #     [-8.48711043, -6.69547573, -8.57844496, 8.10534579],
        #     [-3.11090424, 10.86656431, 4.62638161, 0.91545875],
        #     [-2.14780202, 10.55232269, 4.6040528, 3.53781334],
        #     [-6.05756703, -4.98331661, -9.08371587, 6.56978675],
        #     [-0.62301172, 9.18886394, 4.89742923, 1.89872377],
        #     [-7.54141366, -6.02767626, -9.6308485, 7.20878647],
        #     [-5.72103161, -7.70079191, -7.87495163, 7.73630384],
        #     [-2.17793419, 9.98983126, 4.1607046, 1.78751071],
        #     [-7.19489644, -6.12114037, -9.61115297, 7.08670431],
        #     [-1.4781981, 9.94556625, 3.80066131, 1.66395731],
        #     [-4.7356831, -6.24619057, -10.86347034, 7.50997723],
        #     [-6.36459192, -6.36632364, -8.32328007, 11.17625441],
        #     [-5.79657595, -5.82630754, -10.21599712, 6.38569788],
        #     [-1.68665271, 7.79344248, 4.84874243, 0.01349956],
        #     [-2.85191214, 8.21200886, 4.47859312, 2.37722054],
        #     [-6.58655472, -7.59446101, -6.97255325, 7.79735584],
        #     [-4.23411546, 8.4519986, 3.62704772, 2.28741702],
        #     [-6.30873668, -5.74454395, -7.88432599, 7.97491417],
        #     [-6.40638957, -6.95293851, -9.68512147, 5.80867569],
        #     [-2.96983639, 10.07140835, 4.98349713, 0.21012953],
        #     [-6.82141847, -8.02307989, -8.4805404, 7.88430744],
        #     [-1.36637481, 9.76621916, 5.43091078, 1.06378223],
        #     [-7.32614214, -6.0237108, -8.62423401, 6.07778414],
        #     [-6.37463991, -6.0143544, -10.03862416, 6.98902168],
        #     [-2.18511365, 8.62920385, 3.96295684, 2.58484597],
        #     [-7.76914162, -7.69591988, -8.91542947, 7.66467489],
        #     [-1.03130358, 8.49601591, 3.83138523, 1.47141264],
        #     [-4.42796884, 8.98777225, 4.70010905, 4.4364118],
        #     [-6.43580776, -6.10547554, -9.76525823, 7.26399756],
        #     [-2.70155859, 9.31583347, 4.60516707, 0.80449165],
        #     [-2.97261532, 8.54855637, 4.88184111, 0.05988944],
        #     [-6.60293639, -6.05292634, -8.82532586, 8.77705699],
        #     [-2.50408417, 8.77969899, 3.22450809, 1.55252436],
        #     [-2.58120774, 10.01781903, 5.00151486, 1.32804993],
        #     [-3.4172217, 7.60198243, 6.10552761, 1.74739338],
        #     [-7.95051969, -6.39763718, -9.06179054, 8.03752341],
        #     [-6.193367, -8.49282546, -9.31025962, 8.41247351],
        #     [-0.92998481, 9.78172086, 4.17040445, 2.51572973],
        #     [-4.05986105, 9.0828491, 3.57757512, 2.44676211],
        #     [-2.41743685, 7.02671721, 4.42020695, 2.33028226],
        #     [-6.81534717, -7.95785437, -9.55363147, 8.00312066],
        #     [-3.49973395, 8.4479884, 4.7395302, 1.46969403],
        #     [-6.06610997, -8.11097391, -8.61086782, 8.63066567],
        #     [-7.3545725, -7.53343883, -7.07287352, 7.72850463],
        #     [-2.44166942, 7.58953794, 4.09549611, 2.08409227],
        #     [-7.36499074, -6.79823545, -6.52366919, 5.45625772],
        #     [-2.62484591, 8.71318243, 3.16135685, 1.25332548],
        #     [-2.90130578, 7.55077118, 4.93599911, 2.23422496],
        #     [-6.62913434, -6.53366138, -9.51835248, 7.55577661],
        #     [-1.10640331, 7.61243507, 5.22673593, 4.16362531],
        #     [-7.79905143, -5.33017519, -9.62158105, 7.0014614],
        #     [-8.1165779, -8.20056621, -8.31638619, 7.62050759],
        #     [-8.07093069, -6.22355598, -9.81300943, 8.11060752],
        #     [-7.14428402, -4.15994043, -8.21266041, 6.46636536],
        #     [-6.70644627, -6.49479221, -9.72218519, 7.47724802],
        # ]
        self.points = [[0, 0, 0, 1]] * 64
        # fmt: off
        # self.labels = [
        #     0, 0, 0, 0, 0, 1, 0, 1,
        #     1, 1, 0, 0, 1, 0, 1, 1,
        #     0, 1, 0, 1, 1, 1, 0, 0,
        #     1, 0, 1, 1, 0, 1, 0, 1,
        #     1, 0, 1, 0, 0, 1, 0, 0,
        #     1, 0, 0, 0, 1, 1, 0, 0,
        #     0, 1, 0, 1, 1, 0, 1, 0,
        #     0, 1, 0, 1, 1, 1, 1, 1
        # ]
        self.labels = [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8
        # self.labels = [0, 1] * 32
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
                # [
                #     [0.08763321, -0.19935564, 0.24326038, 0.06141703],
                #     [-0.08763321, 0.19935564, -0.24326038, -0.06141703],
                # ]
                [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0],]
            )
        )
        weights.requires_grad = True
        self.layers.weight = weights

    def forward(self, batch):
        """Docs"""
        return self.layers(batch)

    def train(self, mode: bool = True):
        assert mode == False, "No changes required"
        super().train(mode)


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
