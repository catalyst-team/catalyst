# flake8: noqa
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl, utils


class BatchOverfitCallbackCheck(dl.Callback):
    def __init__(self):
        super().__init__(order=dl.CallbackOrder.external)

    def on_loader_start(self, runner):
        # 320 samples with 32 batch size
        # -> 1 batch size = 32
        # -> 0.1 portion = 32
        assert len(runner.loaders[runner.loader_key]) == 32


def _prepare_experiment():
    # data
    utils.set_global_seed(42)
    num_samples, num_features = int(32e1), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])
    return loaders, model, criterion, optimizer, scheduler


def test_batch_overfit():
    loaders, model, criterion, optimizer, scheduler = _prepare_experiment()
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logs/batch_overfit",
        num_epochs=1,
        verbose=False,
        callbacks=[dl.BatchOverfitCallback(train=1, valid=0.1)],
    )
    assert runner.epoch_metrics["train"]["loss"] < 1.4
    assert runner.epoch_metrics["valid"]["loss"] < 1.3
