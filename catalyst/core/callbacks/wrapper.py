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
        filter_fn: Union[str, Callable[[str, int, str], bool]] = None,
        use_global_epochs: bool = False,
    ):
        """``loaders`` and ``filter_fn`` are interchangeable arguments.

        Args:
            base_callback: callback to wrap
            loaders: loaders to change base callback behaviour
            filter_fn: function to use instead of loaders
        """
        callback = base_callback

        super().__init__(
            order=callback.order, node=callback.node, scope=callback.scope
        )

        self.callback = callback
        self.use_global_epochs = use_global_epochs

        # loader parameters
        self.filter_fn = None
        self._is_active = True

        if loaders is not None:
            if isinstance(loaders, str):
                loaders = [loaders]
            # sequence of loaders
            if isinstance(loaders, (list, tuple)):
                loaders = sorted(set(loaders))  # ignore duplicates
                self.filter_fn = lambda stage, epoch, loader: loader in loaders
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
                self.filter_fn = lambda stage, epoch, loader: epoch in (
                    ignore_list.get(loader) or {}
                )
            else:
                raise ValueError(
                    "'loaders' type should be one of - str, "
                    "Sequence[str], Mapping[str, int] or "
                    "Mapping[str, Sequence[int]]!"
                )
        elif filter_fn is not None:
            if isinstance(filter_fn, str):
                # python lambda functions in config api
                try:
                    filter_fn = eval(filter_fn)
                except (ValueError, SyntaxError):
                    raise ValueError(
                        "'filter_fn' should be a valid "
                        "python lambda function with "
                        "three arguments - 'stage', 'epoch' and 'loader'!"
                    )
            if not callable(filter_fn):
                raise ValueError("'filter_fn' should be a callable!")
            if filter_fn.__code__.co_argcount != 3:
                raise ValueError(
                    "Filter function should have three arguments - "
                    "'stage', 'epoch' and 'loader'!"
                )
            self.filter_fn = filter_fn

    def on_loader_start(self, runner: IRunner) -> None:
        """
        Check if current epoch should be skipped.

        Args:
            runner (IRunner): current runner
        """
        stage = runner.stage_name
        loader = runner.loader_name
        epoch = runner.global_epoch if self.use_global_epochs else runner.epoch

        if self.filter_fn is not None:
            self._is_active = not self.filter_fn(stage, epoch, loader)

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
