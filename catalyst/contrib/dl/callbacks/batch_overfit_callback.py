from collections import OrderedDict
import copy

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.data.loader import BatchLimitLoaderWrapper


class BatchOverfitCallback(Callback):
    """Callback for ovefitting loaders with specified number of batches.
    By default we use ``1`` batch for loader.

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

    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: loader names and their run periods.
        """
        super().__init__(order=CallbackOrder.internal)

        self.loader_batches = {}
        for loader, num_batches in kwargs.items():
            if not isinstance(num_batches, (int, float)):
                raise TypeError(
                    "Expected loader num_batches type is int/float "
                    f"but got {type(num_batches)}"
                )
            self.loader_batches[loader] = num_batches

    def on_epoch_start(self, runner: IRunner) -> None:
        """
        Set loaders for current epoch.
        If validation is not required then the first loader
        from loaders used in current epoch will be used
        as validation loader.
        Metrics from the latest epoch with true
        validation loader will be used
        in the epochs where this loader is missing.

        Arguments:
            runner (IRunner): current runner

        Raises:
            ValueError: if there are no loaders in epoch
        """
        epoch_loaders = OrderedDict()

        for name, loader in runner.loaders.items():
            num_batches = self.loader_batches.get(name, 1)
            if isinstance(num_batches, float):
                num_batches = int(len(loader) * num_batches)
            epoch_loaders[name] = BatchLimitLoaderWrapper(
                loader=loader,
                num_batches=num_batches,
            )

        runner.loaders = epoch_loaders

    def on_epoch_end(self, runner: IRunner):
        runner.loaders = {
            key: value.loader for key, value in runner.loaders.items()
        }


__all__ = ["BatchOverfitCallback"]
