from typing import Callable, Mapping, Sequence, TYPE_CHECKING, Union
from collections import OrderedDict

from catalyst.core.callback import Callback, CallbackWrapper

LOADERS = Union[str, Sequence[str], Mapping[str, Union[int, Sequence[int]]]]
FILTER_FN = Callable[[str, int, str], bool]

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class _EpochFilterFn:
    def __init__(self, epochs: Union[int, float, Sequence[int]], reverse_condition: bool):
        if not isinstance(epochs, (int, float, list, tuple)):
            raise ValueError(
                "'epochs' should be int/float/Sequence[int]! " f"(got {type(epochs)})"
            )
        self.epochs = epochs
        self.reverse_condition = reverse_condition

        # extra conditions precomputing
        if isinstance(self.epochs, (int, float)):
            self.epochs = int(self.epochs)
        elif isinstance(self.epochs, (list, tuple)):
            self.epochs = sorted(set(self.epochs))

    def __call__(self, stage, epoch, loader):
        if isinstance(self.epochs, (int, float)):
            if self.reverse_condition:
                return epoch % self.epochs != 0
            else:
                return epoch % self.epochs == 0
        elif isinstance(self.epochs, (list, tuple)):
            if self.reverse_condition:
                return epoch not in self.epochs
            else:
                return epoch in self.epochs


class _LoaderFilterFn:
    def __init__(self, loaders: LOADERS, reverse_condition: bool):
        if isinstance(loaders, str):
            loaders = [loaders]
        if not isinstance(loaders, (list, tuple, dict, OrderedDict)):
            raise ValueError(
                "'loaders' type should be one of - str, "
                "Sequence[str], Mapping[str, int] or "
                "Mapping[str, Sequence[int]]! "
                f"(got {type(loaders)})"
            )
        self.loaders = loaders
        self.reverse_condition = reverse_condition

        # extra conditions precomputing
        if isinstance(self.loaders, (list, tuple)):
            self.loaders = sorted(set(self.loaders))  # ignore duplicates
        elif isinstance(self.loaders, (dict, OrderedDict)):
            ignore_list = {}
            for loader, epochs in self.loaders.items():
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
            self._ignore_list = ignore_list

    def __call__(self, stage, epoch, loader):
        # sequence of loaders
        if isinstance(self.loaders, (list, tuple)):
            if self.reverse_condition:
                return loader not in self.loaders
            else:
                return loader in self.loaders
        # loader: ignore epoch or epochs
        elif isinstance(self.loaders, (dict, OrderedDict)):
            if self.reverse_condition:
                return epoch not in (
                    self._ignore_list.get(loader) or {}  # {loader: [epoch]}.get(loader)
                )
            else:
                return epoch in (self._ignore_list.get(loader) or {})


class _ArgsFilterFn:
    def __init__(self, filter_fn: Union[str, FILTER_FN]):
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
                "Filter function should have three arguments - " "'stage', 'epoch' and 'loader'!"
            )
        self.filter_fn = filter_fn

    def __call__(self, stage, epoch, loader):
        return self.filter_fn(stage, epoch, loader)


class ControlFlowCallback(CallbackWrapper):
    """Enable/disable callback execution on different
    stages, loaders and epochs.

    Args:
        base_callback: callback to wrap
        epochs: epochs where
            need to **enable** callback, on other epochs
            callback will be disabled.

            If passed int/float then callback will be enabled
            with period specified as epochs value
            (epochs expression ``epoch_number % epochs == 0``)
            and disabled on other epochs.

            If passed list of epochs then will be executed callback
            on specified epochs.

            Default value is ``None``.
        ignore_epochs:: epochs where
            need to **disable** callback, on other epochs
            callback will be enabled.

            If passed int/float then callback will be disabled
            with period specified as epochs value
            (epochs expression ``epoch_number % epochs != 0``)
            and enabled on other epochs.

            If passed list of epochs then will be disabled callback
            on specified epochs.

            Default value is ``None``.
        loaders (str/Sequence[str]/Mapping[str, int/Sequence[str]]):
            loaders where should be **enabled** callback, on
            other loaders callback will be disabled.

            If passed string object then will be disabled callback for
            loader with specified name.

            If passed list/tuple of strings then will be disabled callback
            for loaders with specified names.

            If passed dictionary where key is a string and values
            int or list of integers then callback will be
            disabled on epochs (dictionary value) for specified
            loader (dictionary key).

            Default value is ``None``.
        ignore_loaders (str/Sequence[str]/Mapping[str, int/Sequence[str]]):
            loader names where should be **disabled** callback, on
            other loaders callback will be enabled.

            If passed string object then will be disabled callback for
            loader with specified name.

            If passed list/tuple of strings then will be disabled callback
            for loaders with specified names.

            If passed dictionary where key is a string and values
            int or list of integers then callback will be
            disabled on epochs (dictionary value) for specified
            loader (dictionary key).

            Default value is ``None``.
        filter_fn (str or Callable[[str, int, str], bool]):
            function to use instead of ``loaders`` or ``epochs`` arguments.

            If the object passed to a ``filter_fn`` is a string
            then it will be interpreted as python code. Expected
            lambda function with three arguments stage name (str),
            epoch number (int), loader name (str) and this function
            should return ``True`` if callback should be enabled
            on some condition.

            If passed callable object then it should accept
            three arguments - stage name (str), epoch number (int),
            loader name (str) and should return ``True`` if callback
            should be enabled on some condition othervise should
            return ``False``.

            Default value is ``None``.

            Examples:

            .. code-block:: python

                # enable callback on all loaders
                # exept "train" loader every 2 epochs
                ControlFlowCallback(
                    ...
                    filter_fn=lambda s, e, l: l != "train" and e % 2 == 0
                    ...
                )
                # or with string equivalent
                ControlFlowCallback(
                    ...
                    filter_fn="lambda s, e, l: l != 'train' and e % 2 == 0"
                    ...
                )

        use_global_epochs: if ``True`` then
            will be used global epochs instead of epochs in
            a stage, the default value is ``False``

    .. note::

        Please run experiment with
        :class:`check option
        <catalyst.core.callbacks.early_stop.CheckRunCallback>`
        to check if everything works as expected with this callback.

    For example, if you don't want to compute loss on a validation
    you can ignore
    :class:`CriterionCallback
    <catalyst.core.callbacks.criterion.CriterionCallback>`,
    for **notebook API** need to wrap callback:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.dl import (
            SupervisedRunner, AccuracyCallback,
            CriterionCallback, ControlFlowCallback,
        )

        num_samples, num_features = 10_000, 10
        n_classes = 10
        X = torch.rand(num_samples, num_features)
        y = torch.randint(0, n_classes, [num_samples])
        loader = DataLoader(TensorDataset(X, y), batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        model = torch.nn.Linear(num_features, n_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        runner = SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=5,
            verbose=False,
            valid_metric="accuracy03",
            minimize_metric=False,
            callbacks=[
                AccuracyCallback(
                    accuracy_args=[1, 3, 5]
                ),
                ControlFlowCallback(
                    base_callback=CriterionCallback(),
                    ignore_loaders="valid"  # or loaders="train"
                )
            ]
        )

    In **config API** need to use ``_wrapper`` argument:

    .. code-block:: yaml

        callbacks_params:
          ...
          loss:
            _wrapper:
               callback: ControlFlowCallback
               ignore_loaders: valid
            callback: CriterionCallback
          ...

    """

    def __init__(
        self,
        base_callback: Callback,
        epochs: Union[int, Sequence[int]] = None,
        ignore_epochs: Union[int, Sequence[int]] = None,
        loaders: LOADERS = None,
        ignore_loaders: LOADERS = None,
        filter_fn: Union[str, FILTER_FN] = None,
        use_global_epochs: bool = False,
    ):
        """Init."""
        required_args = (
            epochs,
            ignore_epochs,
            loaders,
            ignore_loaders,
            filter_fn,
        )
        if all(arg is None for arg in required_args):
            raise ValueError(
                "Expected one of arguments - "
                "'epochs', 'ignore_epochs', "
                "'loaders', 'ignore_loaders' "
                "or 'filter_fn'!"
            )

        super().__init__(base_callback, True)
        self.use_global_epochs = use_global_epochs
        # loader parameters
        self.filter_fn = None

        # due to ddp-setup, we have to wrap everything with classes
        if epochs is not None:
            self.filter_fn = _EpochFilterFn(epochs, False)
        elif ignore_epochs is not None:
            self.filter_fn = _EpochFilterFn(ignore_epochs, True)
        elif loaders is not None:
            self.filter_fn = _LoaderFilterFn(loaders, False)
        elif ignore_loaders is not None:
            self.filter_fn = _LoaderFilterFn(ignore_loaders, True)
        elif filter_fn is not None:
            self.filter_fn = _ArgsFilterFn(filter_fn)

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Check if current epoch should be skipped.

        Args:
            runner: current runner
        """
        stage = runner.stage_key
        loader = runner.loader_key
        epoch = runner.global_epoch_step if self.use_global_epochs else runner.stage_epoch_step

        if self.filter_fn is not None:
            self._is_enabled = self.filter_fn(stage, epoch, loader)

        if self._is_enabled:
            self.callback.on_loader_start(runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """
        Reset status of callback

        Args:
            runner: current runner
        """
        if self._is_enabled:
            self.callback.on_loader_end(runner)
        self._is_enabled = True


__all__ = ["ControlFlowCallback"]
