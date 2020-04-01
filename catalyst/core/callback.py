from typing import TYPE_CHECKING
from enum import IntFlag

if TYPE_CHECKING:
    from .state import State


class CallbackNode(IntFlag):
    """Callback node usage flag during distributed training.

    - All (0) - use on all nodes, botch master and worker.
    - Master (1) - use only on master node.
    - Worker (2) - use only in worker nodes.
    """

    All = 0
    Master = 1
    Worker = 2


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

    Internal = 0  # pytorch
    Metric = 20  # pytorch
    MetricAggregation = 40  # pytorch
    Optimizer = 60  # pytorch
    Validation = 80  # numpy
    Scheduler = 100  # numpy
    Logging = 120  # numpy
    External = 200  # numpy


class CallbackScope(IntFlag):
    """Callback scope usage flag during training.

    - Stage (0) - use Callback only during one experiment stage.
    - Experiment (1) - use Callback during whole experiment run.
    """

    Stage = 0
    Experiment = 1


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

            - :py:mod:`catalyst.core.experiment._Experiment`
            - :py:mod:`catalyst.core.runner._Runner`
            - :py:mod:`catalyst.core.state.State`
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
        node: int = CallbackNode.All,
        scope: int = CallbackScope.Stage,
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

    def on_stage_start(self, state: "State"):
        """Event handler for stage start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_stage_end(self, state: "State"):
        """Event handler for stage end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_epoch_start(self, state: "State"):
        """Event handler for epoch start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_epoch_end(self, state: "State"):
        """Event handler for epoch end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_loader_start(self, state: "State"):
        """Event handler for loader start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_loader_end(self, state: "State"):
        """Event handler for loader end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_batch_start(self, state: "State"):
        """Event handler for batch start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_batch_end(self, state: "State"):
        """Event handler for batch end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_exception(self, state: "State"):
        """Event handler for exception case.

        Args:
            state ("State"): State instance.
        """
        pass


__all__ = [
    "Callback",
    "CallbackNode",
    "CallbackOrder",
    "CallbackScope",
]
