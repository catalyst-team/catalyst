from typing import TYPE_CHECKING
from enum import IntFlag

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class CallbackNode(IntFlag):
    """Callback node usage flag during distributed training.

    - All (0) - use on all nodes, botch master and worker.
    - Master (1) - use only on master node.
    - Worker (2) - use only in worker nodes.
    """

    All = all = 0  # noqa: WPS115
    Master = master = 1  # noqa: WPS115
    Worker = worker = 2  # noqa: WPS115


class CallbackOrder(IntFlag):
    """Callback usage order during training.

    Catalyst executes Callbacks with low `CallbackOrder`
    **before** Callbacks with high `CallbackOrder`.

    Predefined orders:

    - **Internal** (0) - some Catalyst Extras,
      like PhaseCallbacks (used in GANs).
    - **Metric** (20) - Callbacks with metrics and losses computation.
    - **MetricAggregation** (40) - metrics aggregation callbacks,
      like sum different losses into one.
    - **Optimizer** (60) - optimizer step,
      requires computed metrics for optimization.
    - **Validation** (80) - validation step,
      computes validation metrics subset based on all metrics.
    - **Scheduler** (100) - scheduler step,
      in `ReduceLROnPlateau` case
      requires computed validation metrics for optimizer schedule.
    - **Logging** (120) - logging step,
      logs metrics to Console/Tensorboard/Alchemy_,
      requires computed metrics.
    - **External** (200) - additional callbacks with custom logic,
      like InferenceCallbacks

    Nevertheless, you always can create CustomCallback with any order,
    for example::

        >>> class MyCustomCallback(Callback):
        >>>     def __init__(self):
        >>>         super().__init__(order=42)
        >>>     ...
        # MyCustomCallback will be executed after all `Metric`-Callbacks
        # but before all `MetricAggregation`-Callbacks.

    .. _Alchemy: https://alchemy.host
    """

    Internal = internal = 0  # noqa: WPS115
    Metric = metric = 20  # noqa: WPS115
    MetricAggregation = metric_aggregation = 40  # noqa: WPS115
    Optimizer = optimizer = 60  # noqa: WPS115
    Validation = validation = 80  # noqa: WPS115
    Scheduler = scheduler = 100  # noqa: WPS115
    Logging = logging = 120  # noqa: WPS115
    External = external = 200  # noqa: WPS115


class CallbackScope(IntFlag):
    """Callback scope usage flag during training.

    - Stage (0) - use Callback only during one experiment stage.
    - Experiment (1) - use Callback during whole experiment run.
    """

    Stage = stage = 0  # noqa: WPS115
    Experiment = experiment = 1  # noqa: WPS115


class Callback:
    """
    An abstraction that lets you customize your experiment run logic.
    To give users maximum flexibility and extensibility Catalyst supports
    callback execution anywhere in the training loop:

    .. code:: bash

        -- stage start
        ---- epoch start
        ------ loader start
        -------- batch start
        ---------- batch handler (Runner logic)
        -------- batch end
        ------ loader end
        ---- epoch end
        -- stage end

        exception – if an Exception was raised

    All callbacks have
        - ``order`` from ``CallbackOrder``
        - ``node`` from ``CallbackNode``
        - ``scope`` from ``CallbackScope``

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment.IExperiment`
            - :py:mod:`catalyst.core.runner.IRunner`
            - :py:mod:`catalyst.core.callback.Callback`

    Abstraction, please check out the implementations:

        - :py:mod:`catalyst.core.callbacks.criterion.CriterionCallback`
        - :py:mod:`catalyst.core.callbacks.optimizer.OptimizerCallback`
        - :py:mod:`catalyst.core.callbacks.scheduler.SchedulerCallback`
        - :py:mod:`catalyst.core.callbacks.logging.TensorboardLogger`
        - :py:mod:`catalyst.core.callbacks.checkpoint.CheckpointCallback`
    """

    def __init__(
        self,
        order: int,
        node: int = CallbackNode.all,
        scope: int = CallbackScope.stage,
    ):
        """Callback initializer.

        Args:
            order: flag from ``CallbackOrder``
            node: flag from ``CallbackNode``
            scope: flag from ``CallbackScope``
        """
        self.node = node
        self.order = order
        self.scope = scope

    def on_stage_start(self, runner: "IRunner"):
        """Event handler for stage start.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_stage_end(self, runner: "IRunner"):
        """Event handler for stage end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_epoch_start(self, runner: "IRunner"):
        """Event handler for epoch start.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_epoch_end(self, runner: "IRunner"):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_loader_start(self, runner: "IRunner"):
        """Event handler for loader start.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_loader_end(self, runner: "IRunner"):
        """Event handler for loader end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_batch_start(self, runner: "IRunner"):
        """Event handler for batch start.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_batch_end(self, runner: "IRunner"):
        """Event handler for batch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass

    def on_exception(self, runner: "IRunner"):
        """Event handler for exception case.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        pass


class WrapperCallback(Callback):
    """Enable/disable callback execution."""

    def __init__(self, base_callback: Callback, enable_callback: bool = True):
        """
        Args:
            base_callback (Callback): callback to wrap
            enable_callback (boolean): indicator to enable/disable
                callback, if ``True`` then callback will be enabled,
                default ``True``
        """
        if base_callback is None or not isinstance(base_callback, Callback):
            raise ValueError(
                f"Expected callback but got - {type(base_callback)}!"
            )
        super().__init__(
            order=base_callback.order,
            node=base_callback.node,
            scope=base_callback.scope,
        )
        self.callback = base_callback
        self._is_enabled = enable_callback

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Check if current epoch should be skipped.

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_loader_start(runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """
        Reset status of callback

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_loader_end(runner)

    def on_stage_start(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_stage_start(runner)

    def on_stage_end(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_stage_end(runner)

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_epoch_start(runner)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_epoch_end(runner)

    def on_batch_start(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_batch_start(runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_batch_end(runner)

    def on_exception(self, runner: "IRunner") -> None:
        """Run base_callback (if possible)

        Args:
            runner (IRunner): current runner
        """
        if self._is_enabled:
            self.callback.on_exception(runner)


__all__ = [
    "Callback",
    "CallbackNode",
    "CallbackOrder",
    "CallbackScope",
    "WrapperCallback",
]
