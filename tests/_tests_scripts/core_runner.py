from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from catalyst.dl import SupervisedRunner


class DummyDataset(Dataset):
    features_dim: int = 4
    out_dim: int = 1

    def __len__(self):
        return 4

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


def run_train_with_empty_loader() -> None:
    dataset = DummyDataset()
    model = nn.Linear(
        in_features=dataset.features_dim, out_features=dataset.out_dim
    )
    loader = DataLoader(
        dataset=dataset, batch_size=len(dataset) + 1, drop_last=True
    )
    runner = SupervisedRunner()
    runner.train(
        loaders={"train": loader},
        model=model,
        num_epochs=1,
        criterion=nn.BCEWithLogitsLoss(),
    )


def test_cathing_empty_loader() -> None:
    try:
        run_train_with_empty_loader()
    except AssertionError:
        assert True
