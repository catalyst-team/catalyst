from typing import Mapping
from collections import OrderedDict
import copy

from torch.utils.data import DataLoader

from catalyst.core import Callback, CallbackOrder, State


class PeriodicLoaderRunnerCallback(Callback):
    """A callback to run validation with some period."""

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: expected loader names and periods to run loader.

        Example:

            >>> PeriodicLoaderRunnerCallback(
            >>>     train_additional=2,
            >>>     valid=3,
            >>>     valid_additional=5
            >>> )
        """
        super().__init__(order=CallbackOrder.External)

        self.valid_loader: str = None
        self.valid_metrics: Mapping[str, float] = None
        self.loaders: Mapping[str, DataLoader] = OrderedDict()

        required_callbacks = {"train"}
        self.loader_periods = {}
        for loader, period in kwargs.items():
            if not isinstance(loader, str) or not isinstance(
                period, (int, float)
            ):
                continue
            if loader in required_callbacks:
                continue
            self.loader_periods[loader] = period

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

    def on_epoch_start(self, state: State) -> None:
        """Set loaders for current epoch.

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
        state.valid_loader = (
            self.valid_loader
            if self.valid_loader in epoch_loaders
            else "train"
        )
        state.loaders = epoch_loaders

    def on_epoch_end(self, state: State) -> None:
        """Store validation metrics and use latest validation score
        if epoch don't have validation dataloader.

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
