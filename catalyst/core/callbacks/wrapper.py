from typing import Any, Callable, Mapping, Sequence, Union
from collections import OrderedDict

from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner
from catalyst.dl import registry

LOADERS = Sequence[str]
LOADERS_WITH_EPOCHS = Mapping[str, Union[int, Sequence[int]]]
LOADERS_LAMBDA = Callable[
    [int, str], bool
]  # lambda epoch, loader: return True


class WrapperCallback(Callback):
    """
    Custumize callback execution on different loaders.

    Examples:

    >>> from catalyst.dl import (
    >>>     SupervisedRunner, FilterCallback, CriterionCallback
    >>> )
    >>> runner = SupervisedRunner()
    >>> runner.train(
    >>>     ...
    >>>     callbacks=[
    >>>         FilterCallback(
    >>>             wrap=CriterionCallback(),
    >>>             loaders=["valid"]
    >>>         ),
    >>>     ]
    >>> )
    """

    def __init__(
        self,
        base_callback: Union[Callback, Mapping[str, Any]],
        loaders: Union[LOADERS, LOADERS_WITH_EPOCHS] = None,
        ignore_foo: Callable[[int, str], bool] = None,
    ):
        """``loaders`` and ``ignore_foo`` are interchangeable arguments.

        Args:
            base_callback: callback to wrap
            loaders: loaders to change base callback behaviour
            ignore_foo: function to use instead of loaders
        """
        if isinstance(base_callback, Callback):
            callback = base_callback
        else:
            callback = registry.CALLBACKS.get_from_params(**base_callback)

        super().__init__(
            order=callback.order, node=callback.node, scope=callback.scope
        )

        self.callback = callback

        # loader parameters
        self.ignore_foo = None
        self._is_active = True

        if loaders is not None:
            if isinstance(loaders, str):
                loaders = [loaders]
            # sequence of loaders
            if isinstance(loaders, (list, tuple)):
                self.ignore_foo = lambda epoch, loader: loader in loaders
            # loader: ignore epoch or epochs
            elif isinstance(loaders, (dict, OrderedDict)):
                ignore_list = {}
                for loader, epochs in loaders.items():
                    if isinstance(epochs, (int, float)):
                        ignore_list[loader] = [int(epochs)]
                    else:
                        ignore_list[loader] = [int(num) for num in epochs]
                self.ignore_foo = lambda epoch, loader: epoch in (
                    ignore_list.get(loader) or {}
                )
        elif ignore_foo is not None:
            self.ignore_foo = ignore_foo

    def on_loader_start(self, runner: IRunner) -> None:
        """
        Check if current epoch should be skipped.

        Args:
            runner (IRunner): current runner
        """
        loader = runner.loader_name
        epoch = runner.epoch

        if self.ignore_foo is not None:
            self._is_active = not self.ignore_foo(epoch, loader)

        # print(f"({loader} {epoch})>> {self._is_active}")

        self.callback.on_loader_start(runner)

    def on_loader_end(self, runner: IRunner) -> None:
        """
        Reset status of callback

        Args:
            runner (IRunner): current runner
        """
        self.callback.on_loader_end(runner)
        self._is_active = True

    def on_stage_start(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_stage_start(runner)

    def on_stage_end(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_stage_end(runner)

    def on_epoch_start(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_epoch_start(runner)

    def on_epoch_end(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_epoch_end(runner)

    def on_batch_start(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_batch_start(runner)

    def on_batch_end(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_batch_end(runner)

    def on_exception(self, runner: IRunner) -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
            self.callback.on_exception(runner)


__all__ = ["WrapperCallback"]
