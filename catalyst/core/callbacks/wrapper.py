from typing import Callable, Mapping, Sequence, Union
from collections import OrderedDict

from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner

LOADERS = Union[str, Sequence[str], Mapping[str, Union[int, Sequence[int]]]]
FILTER_FN = Callable[[str, int, str], bool]


def _filter_fn_from_loaders(loaders: LOADERS) -> FILTER_FN:
    """Build ``filter_fn`` from loaders for `WrapperCallback`.

    Args:
        loaders (str/Sequence[str]/Mapping[str, int/Sequence[str]]):
            loaders description

    Raises:
        ValueError: if can't build filter_fn from mappings
        ValueError: if passed object with unexpected type

    Returns:
        filter function which accepts 3 arguments - stage (str),
        epoch (int), loader (str) and return ``True`` if
        need to disable callback
    """
    if isinstance(loaders, str):
        loaders = [loaders]

    # sequence of loaders
    if isinstance(loaders, (list, tuple)):
        loaders = sorted(set(loaders))  # ignore duplicates
        filter_fn = lambda stage, epoch, loader: loader in loaders
    # loader: ignore epoch or epochs
    elif isinstance(loaders, (dict, OrderedDict)):
        ignore_list = {}
        for loader, epochs in loaders.items():
            if isinstance(epochs, (int, float)):
                ignore_list[loader] = [int(epochs)]
            else:
                try:
                    ignore_list[loader] = []
                    for num in sorted(set(epochs)):
                        to_add = int(num)
                        ignore_list[loader].append(to_add)
                except (ValueError, TypeError):
                    raise ValueError(
                        "'ignore_list' should be a dict where "
                        "keys is a int/float/List[int]/Tuple[int]!"
                    )
        filter_fn = lambda stage, epoch, loader: epoch in (
            ignore_list.get(loader) or {}
        )
    else:
        raise ValueError(
            "'loaders' type should be one of - str, "
            "Sequence[str], Mapping[str, int] or "
            "Mapping[str, Sequence[int]]!"
        )
    return filter_fn


def _filter_fn_from_arg(filter_fn: Union[str, FILTER_FN]) -> FILTER_FN:
    """Check if filter function from argumets
    can be used with filter callback.

    Args:
        filter_fn (str or Callable): filter function to check

    Raises:
        ValueError: if ``filter_fn`` is a string and can not be
            interpreted as python code then an error will be raised
        ValueError: if passed not callable object then will be
            raised an error
        ValueError: will be raised error if filter function do not
            have three arguments

    Returns:
        filter function which accepts 3 arguments - stage (str),
        epoch (int), loader (str) and return ``True`` if
        need to disable callback
    """
    if isinstance(filter_fn, str):
        # lambda function from string
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
    return filter_fn


class WrapperCallback(Callback):
    """
    Customize callback execution on different stages, loaders and epoch.

    For example, if you don't want to compute loss on a validation
    you can ignore ``CriterionCallback``, for notebook API
    need to wrap callback:

    .. code-block:: python

        from catalyst.dl import (
            SupervisedRunner, WrapperCallback, CriterionCallback
        )
        runner = SupervisedRunner()
        runner.train(
            ...
            callbacks=[
                ...
                WrapperCallback(
                    base_callback=CriterionCallback(),
                    loaders=["valid"]
                ),
                ...
            ]
            ...
        )

    In config API need to use ``_wrapper`` argument:

    .. code-block:: yaml

        callbacks_params:
          ...
          loss:
            _wrapper:
               callback: WrapperCallback
               loaders: valid
            callback: CriterionCallback
          ...

    """

    def __init__(
        self,
        base_callback: Callback,
        loaders: LOADERS = None,
        filter_fn: Union[str, FILTER_FN] = None,
        use_global_epochs: bool = False,
    ):
        """
        Args:
            base_callback (Callback): callback to wrap
            loaders (str/Sequence[str]/Mapping[str, int/Sequence[str]]):
                loaders to change base callback behaviour.

                If passed string object then will be ignored loader
                with this name.

                If passed list/tuple of strings then will be ignored
                loaders with same names.

                If passed dictionary where key is a string and values
                int or list of integers then will be ignored epochs
                (dictionary value) for loader (dictionary key).

                Default value is ``None``.
            filter_fn (str or Callable[[str, int, str], bool]):
                function to use instead of loaders.

                If the object passed to a ``filter_fn`` is a string
                then it will be evaluated as python code
                with a lambda function.

                If passed callable object then it should accept
                3 arguments - stage name (str), epoch number (int),
                loader name (str) and return ``True`` (if callback
                should be ignored) or ``False``.

                Default value is ``None``.
            use_global_epochs (bool): if ``True`` then
                will be used global epochs instead of epochs in
                a stage, the default value is ``False``
        """
        if loaders is None and filter_fn is None:
            raise ValueError(
                "Expected one of arguments - 'loaders' or 'filter_fn'!"
            )

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
            self.filter_fn = _filter_fn_from_loaders(loaders)
        elif filter_fn is not None:
            self.filter_fn = _filter_fn_from_arg(filter_fn)

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
