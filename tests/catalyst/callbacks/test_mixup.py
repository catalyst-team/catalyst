# flake8: noqa
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl, utils
from catalyst.callbacks import MixupCallback


class DymmyRunner(dl.Runner):
    def handle_batch(self, batch: Tuple[torch.Tensor]):
        self.batch = {"image": batch[0], "clf_targets_one_hot": batch[1]}


def test_mixup_1():
    utils.set_global_seed(42)
    num_samples, num_features, num_classes = int(1e4), int(1e1), 4
    X = torch.rand(num_samples, num_features)
    y = (torch.rand(num_samples) * num_classes).to(torch.int64)
    y = torch.nn.functional.one_hot(y, num_classes).double()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}
    runner = DymmyRunner()
    callback = MixupCallback(keys=["image", "clf_targets_one_hot"])
    for loader_name in ["train", "valid"]:
        for batch in loaders[loader_name]:
            runner.handle_batch(batch)
            callback.on_batch_start(runner)
            assert runner.batch["clf_targets_one_hot"].max(1)[0].mean() < 1
