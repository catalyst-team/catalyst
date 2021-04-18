# flake8: noqa
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.data.loader import BatchLimitLoaderWrapper


def test_batch_limit1() -> None:
    for shuffle in (False, True):
        num_samples, num_features = int(1e2), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=shuffle)
        loader = BatchLimitLoaderWrapper(loader, num_batches=1)

        batch1 = next(iter(loader))[0]
        batch2 = next(iter(loader))[0]
        batch3 = next(iter(loader))[0]
        assert all(torch.isclose(x, y).all() for x, y in zip(batch1, batch2))
        assert all(torch.isclose(x, y).all() for x, y in zip(batch2, batch3))


def test_batch_limit2() -> None:
    for shuffle in (False, True):
        num_samples, num_features = int(1e2), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=shuffle)
        loader = BatchLimitLoaderWrapper(loader, num_batches=2)

        batch1 = next(iter(loader))[0]
        batch2 = next(iter(loader))[0]
        batch3 = next(iter(loader))[0]
        batch4 = next(iter(loader))[0]
        assert all(torch.isclose(x, y).all() for x, y in zip(batch1, batch3))
        assert all(torch.isclose(x, y).all() for x, y in zip(batch2, batch4))
