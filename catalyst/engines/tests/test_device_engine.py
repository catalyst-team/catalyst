from pytest import mark

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from catalyst.callbacks import CriterionCallback
from catalyst.engines import DeviceEngine
from catalyst.dl import SupervisedRunner
from catalyst.experiments import Experiment
from catalyst.settings import IS_CUDA_AVAILABLE

NUM_CUDA_DEVICES = torch.cuda.device_count()


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


def run_train_with_device(device):
    dataset = DummyDataset(10)
    loader = DataLoader(dataset, batch_size=4)
    runner = SupervisedRunner()
    exp = Experiment(
        model=_model_fn,
        loaders={"train": loader, "valid": loader},
        criterion=nn.MSELoss,
        main_metric="loss",
        callbacks=[CriterionCallback()],
        engine=DeviceEngine(device),
    )
    runner.run_experiment(exp)


def test_engine_with_cpu():
    run_train_with_device("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_engine_with_cuda():
    run_train_with_device("cuda")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2,
    reason="Number of CUDA devices is less than 2",
)
def test_engine_with_another_cuda_device():
    run_train_with_device("cuda:1")
