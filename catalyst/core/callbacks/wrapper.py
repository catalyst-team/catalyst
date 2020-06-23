from typing import Callable, Mapping, Sequence, Union
from collections import OrderedDict

from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner


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
        base_callback: Callback,
        loaders: Union[
            str, Sequence[str], Mapping[str, Union[int, Sequence[int]]]
        ] = None,
        ignore_foo: Callable[[int, str], bool] = None,
    ):
        """``loaders`` and ``ignore_foo`` are interchangeable arguments.

        Args:
            base_callback: callback to wrap
            loaders: loaders to change base callback behaviour
            ignore_foo: function to use instead of loaders
        """
        callback = base_callback

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
                loaders = sorted(set(loaders))  # ignore duplicates
                self.ignore_foo = lambda epoch, loader: loader in loaders
            # loader: ignore epoch or epochs
            elif isinstance(loaders, (dict, OrderedDict)):
                ignore_list = {}
                for loader, epochs in loaders.items():
                    if isinstance(epochs, (int, float)):
                        ignore_list[loader] = [int(epochs)]
                    elif isinstance(epochs, (list, tuple)):
                        ignore_list[loader] = []
                        for num in sorted(set(epochs)):
                            try:
                                to_add = int(num)
                                ignore_list[loader].append(to_add)
                            except (ValueError, TypeError):
                                raise ValueError(
                                    "'ignore_list' should be a dict where "
                                    "keys is a int/float/List[int]/Tuple[int]!"
                                )
                    else:
                        raise ValueError(
                            "'ignore_list' should be a dict where "
                            "keys is a int/float/List[int]/Tuple[int]!"
                        )
                self.ignore_foo = lambda epoch, loader: epoch in (
                    ignore_list.get(loader) or {}
                )
            else:
                raise ValueError(
                    "'loaders' type should be one of - str, "
                    "Sequence[str], Mapping[str, int] or "
                    "Mapping[str, Sequence[int]]!"
                )
        elif ignore_foo is not None:
            if not callable(ignore_foo):
                raise ValueError("'ignore_foo' should be a callable!")
            if ignore_foo.__code__.co_argcount != 2:
                raise ValueError(
                    "Ignore function should have two arguments - "
                    "'epoch' and 'loader'!"
                )
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

        if self._is_active:
            self.callback.on_loader_start(runner)

    def on_loader_end(self, runner: IRunner) -> None:
        """
        Reset status of callback

        Args:
            runner (IRunner): current runner
        """
        if self._is_active:
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
