from typing import Mapping
from collections import OrderedDict
import copy

from torch.utils.data import DataLoader

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner


class PeriodicLoaderCallback(Callback):
    """Callback for runing loaders with specified period.
    To disable loader use ``0`` as period (if specified
    ``0`` for validation loader then will be raised an
    error).


    For example, if you have ``train``, ``train_additional``,
    ``valid`` and ``valid_additional`` loaders and wan't to
    use ``train_additional`` every 2 epochs, ``valid`` - every
    3 epochs and ``valid_additional`` - every 5 epochs:

    .. code-block:: python

        from catalyst.dl import (
            SupervisedRunner, PeriodicLoaderRunnerCallback,
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
                PeriodicLoaderRunnerCallback(
                    train_additional=2,
                    valid=3,
                    valid_additional=5
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
        super().__init__(order=CallbackOrder.validation - 1)

        self.valid_loader: str = None
        self.loaders: Mapping[str, DataLoader] = OrderedDict()

        self.loader_periods = {}
        for loader, period in kwargs.items():
            if not isinstance(period, (int, float)):
                raise TypeError(
                    "Expected loader period type is int/float "
                    f"but got {type(period)}!"
                )
            period = int(period)
            if period < 0:
                raise ValueError(f"Period should be >= 0, but got - {period}!")
            self.loader_periods[loader] = period

    def on_stage_start(self, runner: IRunner) -> None:
        """Collect information about loaders.

        Args:
            runner (IRunner): current runner

        Raises:
            ValueError: if there are no loaders in epoch
        """
        # store pointers to data loader objects
        for name, loader in runner.loaders.items():
            self.loaders[name] = loader
        # stage validation loader
        self.valid_loader = copy.copy(runner.valid_loader)
        is_loaders_match = all(
            loader in runner.loaders for loader in self.loader_periods.keys()
        )
        is_same_loaders_number = len(self.loader_periods) == len(
            runner.loaders
        )
        if is_same_loaders_number and is_loaders_match:
            # find potential epoch with zero loaders
            zero_loaders_epochs = list(
                filter(
                    lambda n: all(
                        (p == 0 or n % p != 0)
                        for p in self.loader_periods.values()
                    ),
                    range(1, runner.num_epochs + 1),
                )
            )
            if len(zero_loaders_epochs) > 0:
                epoch_with_err = zero_loaders_epochs[0]
                raise ValueError(
                    f"There will be no loaders in epoch {epoch_with_err}!"
                )

        if self.loader_periods.get(runner.valid_loader, 1) < 1:
            raise ValueError(
                f"Period for a validation loader ('{runner.valid_loader}') "
                "should be > 0!"
            )

    def on_epoch_start(self, runner: IRunner) -> None:
        """
        Set loaders for current epoch.
        If validation is not required then the first loader
        from loaders used in current epoch will be used
        as validation loader.
        Metrics from the latest epoch with true
        validation loader will be used
        in the epochs where this loader is missing.

        Args:
            runner (IRunner): current runner

        Raises:
            ValueError: if there are no loaders in epoch
        """
        epoch_num = runner.epoch
        # loaders to use in current epoch
        epoch_loaders = OrderedDict()
        for name, loader in self.loaders.items():
            period = self.loader_periods.get(name, 1)
            # ignore loaders where period - 0
            if period > 0 and epoch_num % period == 0:
                epoch_loaders[name] = loader
        if len(epoch_loaders) == 0:
            raise ValueError(f"There is no loaders in epoch {epoch_num}!")
        first_loader = next(iter(epoch_loaders.keys()))
        runner.valid_loader = (
            self.valid_loader
            if self.valid_loader in epoch_loaders
            else first_loader
        )
        runner.loaders = epoch_loaders

    def on_epoch_end(self, runner: IRunner) -> None:
        """Check if validation metric should be
        dropped for current epoch.

        Args:
            runner (IRunner): current runner
        """
        valid_metric_name = f"{runner.valid_loader}_{runner.main_metric}"
        if self.valid_loader not in runner.loaders:
            runner.epoch_metrics[valid_metric_name] = (
                float("+inf") if runner.minimize_metric else float("-inf")
            )


__all__ = ["PeriodicLoaderCallback"]
