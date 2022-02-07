from typing import TYPE_CHECKING
from enum import IntFlag

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ICallback:
    """A callable abstraction for deep learning runs."""

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler for experiment start."""
        pass

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Event handler for epoch start."""
        pass

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler for loader start."""
        pass

    def on_batch_start(self, runner: "IRunner") -> None:
        """Event handler for batch start."""
        pass

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler for batch end."""
        pass

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler for loader end."""
        pass

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler for epoch end."""
        pass

    def on_experiment_end(self, runner: "IRunner") -> None:
        """Event handler for experiment end."""
        pass

    def on_exception(self, runner: "IRunner") -> None:
        """Event handler for exception case."""
        pass


class CallbackOrder(IntFlag):
    """Callback usage order during training.

    Catalyst executes Callbacks with low `CallbackOrder`
    **before** Callbacks with high `CallbackOrder`.

    Predefined orders:

    - **Internal** (0) - some Catalyst Extras,
      like PhaseCallbacks (used in GANs).
    - **Metric** (10) - Callbacks with metrics and losses computation.
    - **MetricAggregation** (20) - metrics aggregation callbacks,
      like sum different losses into one.
    - **Backward** (30) - backward step.
    - **Optimizer** (40) - optimizer step,
      requires computed metrics for optimization.
    - **Scheduler** (50) - scheduler step,
      in `ReduceLROnPlateau` case
      requires computed validation metrics for optimizer schedule.
    - **Checkpoint** (60) - checkpoint step.
    - **External** (100) - additional callbacks with custom logic.

    Nevertheless, you always can create CustomCallback with any order,
    for example::

        >>> class MyCustomCallback(Callback):
        >>>     def __init__(self):
        >>>         super().__init__(order=13)
        >>>     ...
        # MyCustomCallback will be executed after all `Metric`-Callbacks
        # but before all `MetricAggregation`-Callbacks.
    """

    Internal = internal = 0
    Metric = metric = 10
    MetricAggregation = metric_aggregation = 20
    Backward = backward = 30
    Optimizer = optimizer = 40
    Scheduler = scheduler = 50
    Checkpoint = checkpoint = 50
    External = external = 100


class Callback(ICallback):
    """
    An abstraction that lets you customize your experiment run logic.

    Args:
        order: flag from ``CallbackOrder``

    To give users maximum flexibility and extensibility Catalyst supports
    callback execution anywhere in the training loop:

    .. code:: bash

        -- experiment start
        ---- epoch start
        ------ loader start
        -------- batch start
        ---------- batch handler (Runner logic)
        -------- batch end
        ------ loader end
        ---- epoch end
        -- experiment end

        exception – if an Exception was raised

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.callbacks.criterion.CriterionCallback`
        - :py:mod:`catalyst.callbacks.optimizer.OptimizerCallback`
        - :py:mod:`catalyst.callbacks.scheduler.SchedulerCallback`
        - :py:mod:`catalyst.callbacks.checkpoint.CheckpointCallback`

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.engine.Engine`
            - :py:mod:`catalyst.core.callback.Callback`

    """

    def __init__(self, order: int):
        """Init."""
        self.order = order


class IMetricCallback(Callback):
    """Metric callback interface, abstraction over metric step."""

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.Metric)


class ICriterionCallback(IMetricCallback):
    """Criterion callback interface, abstraction over criterion step."""

    pass


class IBackwardCallback(Callback):
    """Backward callback interface, abstraction over backward step."""

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.Backward)


class IOptimizerCallback(Callback):
    """Optimizer callback interface, abstraction over optimizer step."""

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.Optimizer)


class ISchedulerCallback(Callback):
    """Scheduler callback interface, abstraction over scheduler step."""

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.Scheduler)


class ICheckpointCallback(Callback):
    """Checkpoint callback interface, abstraction over checkpoint step."""

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.Checkpoint)


class CallbackWrapper(Callback):
    """Enable/disable callback execution.

    Args:
        base_callback: callback to wrap
        enable_callback: indicator to enable/disable
            callback, if ``True`` then callback will be enabled,
            default ``True``
    """

    def __init__(self, base_callback: Callback, enable_callback: bool = True):
        """Init."""
        if base_callback is None or not isinstance(base_callback, Callback):
            raise ValueError(f"Expected callback but got - {type(base_callback)}!")
        super().__init__(order=base_callback.order)
        self.callback = base_callback
        self._is_enabled = enable_callback

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler for experiment start."""
        if self._is_enabled:
            self.callback.on_experiment_start(runner)

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Event handler for epoch start."""
        if self._is_enabled:
            self.callback.on_epoch_start(runner)

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler for loader start."""
        if self._is_enabled:
            self.callback.on_loader_start(runner)

    def on_batch_start(self, runner: "IRunner") -> None:
        """Event handler for batch start."""
        if self._is_enabled:
            self.callback.on_batch_start(runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler for batch end."""
        if self._is_enabled:
            self.callback.on_batch_end(runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler for loader end."""
        if self._is_enabled:
            self.callback.on_loader_end(runner)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler for epoch end."""
        if self._is_enabled:
            self.callback.on_epoch_end(runner)

    def on_experiment_end(self, runner: "IRunner") -> None:
        """Event handler for experiment end."""
        if self._is_enabled:
            self.callback.on_experiment_end(runner)

    def on_exception(self, runner: "IRunner") -> None:
        """Event handler for exception case."""
        if self._is_enabled:
            self.callback.on_exception(runner)


__all__ = [
    "ICallback",
    "Callback",
    "CallbackOrder",
    "IMetricCallback",
    "ICriterionCallback",
    "IBackwardCallback",
    "IOptimizerCallback",
    "ISchedulerCallback",
    "ICheckpointCallback",
    "CallbackWrapper",
]
