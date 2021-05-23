from typing import TYPE_CHECKING
from collections import OrderedDict

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.data.loader import BatchLimitLoaderWrapper

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class BatchOverfitCallback(Callback):
    """Callback to overfit loaders with specified number of batches.
    By default we use ``1`` batch for loader.

    Args:
        kwargs: loader names and their number of batches to overfit.

    For example, if you have ``train``, ``train_additional``,
    ``valid`` and ``valid_additional`` loaders and wan't to overfit
    ``train`` on first 1 batch,
    ``train_additional`` on first 2 batches,
    ``valid`` - on first 20% of batches
    and ``valid_additional`` - on 50% batches:

    .. code-block:: python

        from catalyst.dl import (
            SupervisedRunner, BatchOverfitCallback,
        )
        runner = SupervisedRunner()
        runner.train(
            ...
            loaders={
                "train": ...,
                "train_additional": ...,
                "valid": ...,
                "valid_additional":...
            }
            ...
            callbacks=[
                ...
                BatchOverfitCallback(
                    train_additional=2,
                    valid=0.2,
                    valid_additional=0.5
                ),
                ...
            ]
            ...
        )

    Minimal working example

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # data
        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=8,
            verbose=True,
            callbacks=[dl.BatchOverfitCallback(train=10, valid=0.5)]
        )

    """

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(order=CallbackOrder.internal)

        self.loader_batches = {}
        for loader, num_batches in kwargs.items():
            if not isinstance(num_batches, (int, float)):
                raise TypeError(
                    "Expected loader num_batches type is int/float " f"but got {type(num_batches)}"
                )
            self.loader_batches[loader] = num_batches

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Wraps loaders for current epoch.
        If number-of-batches for loader is not provided then the first batch
        from loader will be used for overfitting.

        Args:
            runner: current runner
        """
        epoch_loaders = OrderedDict()

        for name, loader in runner.loaders.items():
            num_batches = self.loader_batches.get(name, 1)
            if isinstance(num_batches, float):
                num_batches = int(len(loader) * num_batches)
            epoch_loaders[name] = BatchLimitLoaderWrapper(loader=loader, num_batches=num_batches)

        runner.loaders = epoch_loaders

    def on_epoch_end(self, runner: "IRunner"):
        """Unwraps loaders for current epoch.

        Args:
            runner: current runner
        """
        runner.loaders = {
            key: value.origin if isinstance(value, BatchLimitLoaderWrapper) else value
            for key, value in runner.loaders.items()
        }


__all__ = ["BatchOverfitCallback"]
