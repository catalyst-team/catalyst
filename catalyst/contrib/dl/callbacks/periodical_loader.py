from typing import Mapping
from collections import OrderedDict
import copy

from torch.utils.data import DataLoader

from catalyst.core import Callback, CallbackOrder, State


class PeriodicLoaderRunnerCallback(Callback):
    """Callback for runing loaders with specified period.
    To disable loader use ``0`` as period.

    Example:

        >>> PeriodicLoaderRunnerCallback(
        >>>     train_additional=2,
        >>>     valid=3,
        >>>     valid_additional=5
        >>> )
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: loader names and their run periods.
        """
        super().__init__(order=CallbackOrder.External)

        self.valid_loader: str = None
        self.valid_metrics: Mapping[str, float] = None
        self.loaders: Mapping[str, DataLoader] = OrderedDict()

        self.loader_periods = {}
        for loader, period in kwargs.items():
            if not isinstance(loader, str):
                raise TypeError(
                    "Expected loader type is string " f"but got {type(loader)}"
                )
            if not isinstance(period, (int, float)):
                raise TypeError(
                    "Expected loader period type is int/float "
                    f"but got {type(period)}"
                )
            self.loader_periods[loader] = int(period)

    def on_stage_start(self, state: State) -> None:
        """Collect information about loaders.

        Arguments:
            state (State): training state
        """
        # store pointers to data loader objects
        for name, loader in state.loaders.items():
            self.loaders[name] = loader
        # stage validation loader
        self.valid_loader = copy.copy(state.valid_loader)
        is_loaders_match = all(
            loader in state.loaders for loader in self.loader_periods.keys()
        )
        is_same_loaders_number = len(self.loader_periods) == len(state.loaders)
        if is_same_loaders_number and is_loaders_match:
            # find potential epoch with zero loaders
            zero_loaders_epochs = list(
                filter(
                    lambda n: all(
                        (p == 0 or n % p != 0) for p in self.loader_periods.values()
                    ),
                    range(1, state.num_epochs + 1),
                )
            )
            if len(zero_loaders_epochs) > 0:
                epoch_with_err = zero_loaders_epochs[0]
                raise ValueError(
                    f"There will be no loaders in epoch {epoch_with_err}!"
                )

    def on_epoch_start(self, state: State) -> None:
        """Set loaders for current epoch.
        If validation is not required then first loader
        from epoch loaders will be used
        as validation loader but validation metrics will
        be populated from latest epoch where validation
        was required.

        Arguments:
            state (State): training state
        """
        epoch_num = state.epoch
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
        state.valid_loader = (
            self.valid_loader
            if self.valid_loader in epoch_loaders
            else first_loader
        )
        state.loaders = epoch_loaders

    def on_epoch_end(self, state: State) -> None:
        """Store validation metrics and use latest validation score
        when validation loader is not required.

        Arguments:
            state (State): training state
        """
        if self.valid_loader in state.loaders:
            self.valid_metrics = {
                state.main_metric: state.valid_metrics[state.main_metric]
            }
        elif self.valid_metrics is not None:
            # use previous score on validation
            state.valid_metrics = self.valid_metrics


__all__ = ["PeriodicLoaderRunnerCallback"]
