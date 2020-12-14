import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from catalyst.engines import DeviceEngine
from catalyst.dl import SupervisedRunner
from catalyst.experiments import Experiment


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


def _model_fn(in_features, out_features):
    return nn.Linear(in_features, out_features)


def run_train_with_engine():
    dataset = DummyDataset(10)
    loader = DataLoader(dataset, batch_size=4)
    runner = SupervisedRunner()
    exp = Experiment(
        model=_model_fn(dataset.features_dim, dataset.out_dim),
        loaders={"train": loader, "valid": loader},
        engine=DeviceEngine("cpu"),
    )
    runner.run_experiment(exp)


def test_work_with_engine():
    run_train_with_engine()
