from typing import TYPE_CHECKING  # isort:skip
from enum import IntFlag

if TYPE_CHECKING:
    from .state import State


class CallbackNode(IntFlag):
    """
    Callback node usage flag during distributed training.

    - All (0) - use on all nodes, botch master and worker.
    - Master (1) - use only on master node.
    - Worker (2) - use only in worker nodes.
    """
    All = 0
    Master = 1
    Worker = 2


class CallbackOrder(IntFlag):
    """
    Callback usage order flag during training.
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
    """
    Callback scope usage flag during training.

    - Stage (0) - use Callback only during one experiment stage.
    - Experiment (1) - use Callback during whole experiment run.
    """
    Stage = 0
    Experiment = 1


class Callback:
    """
    Abstract class that all callback (e.g., Metrics, Logger)
    classes extends from. Must be extended before usage.

    usage example:

    .. code:: bash

        -- stage start
        ---- epoch start (one epoch - one run of every loader)
        ------ loader start
        -------- batch start
        -------- batch handler
        -------- batch end
        ------ loader end
        ---- epoch end
        -- stage end

        exception – if an Exception was raised

    All callbacks have
        - ``order`` value from ``CallbackOrder``
        - ``node`` value from ``CallbackNode``
        - ``scope`` value from ``CallbackScope``
    """
    def __init__(
        self,
        order: int,
        node: int = CallbackNode.All,
        scope: int = CallbackScope.Stage,
    ):
        """
        Callback initializer.

        Args:
            order: flag from  ``CallbackOrder``
            node: flag from  ``CallbackNode``
            scope: flag from  ``CallbackScope``
        """
        self.node = node
        self.order = order
        self.scope = scope

    def on_stage_start(self, state: "State"):
        """
        Event handler for stage start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_stage_end(self, state: "State"):
        """
         Event handler for stage end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_epoch_start(self, state: "State"):
        """
        Event handler for epoch start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_epoch_end(self, state: "State"):
        """
         Event handler for epoch end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_loader_start(self, state: "State"):
        """
        Event handler for loader start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_loader_end(self, state: "State"):
        """
         Event handler for loader end.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_batch_start(self, state: "State"):
        """
        Event handler for batch start.

        Args:
            state ("State"): State instance.
        """
        pass

    def on_batch_end(self, state: "State"):
        """
        Event handler for batch end.

       Args:
           state ("State"): State instance.
       """
        pass

    def on_exception(self, state: "State"):
        """
        Event handler for exception case.

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
