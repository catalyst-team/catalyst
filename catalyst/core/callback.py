from typing import TYPE_CHECKING  # isort:skip
from enum import IntFlag

if TYPE_CHECKING:
    from .state import _State


class CallbackOrder(IntFlag):
    Internal = 0  # pytorch
    Metric = 20  # pytorch
    MetricAggregation = 40  # pytorch
    Optimizer = 60  # pytorch
    Validation = 80  # numpy
    Logging = 100  # numpy
    Scheduler = 120  # numpy
    External = 200  # numpy


class CallbackNode(IntFlag):
    All = 0
    Master = 1
    Worker = 2


class CallbackType(IntFlag):
    Stage = 0
    Experiment = 1


class Callback:
    """
    Abstract class that all callback (e.g., Logger) classes extends from.
    Must be extended before usage.

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

    All callbacks has ``order`` value from ``CallbackOrder``
    and ``node`` value from ``CallbackNode``
    """
    def __init__(
        self,
        order: int,
        node: int = CallbackNode.All,
        type: int = CallbackType.Stage,
    ):
        """
        For order see ``CallbackOrder`` class
        """
        self.order = order
        self.node = node
        self.type = type

    def on_stage_start(self, state: "_State"):
        pass

    def on_stage_end(self, state: "_State"):
        pass

    def on_epoch_start(self, state: "_State"):
        pass

    def on_epoch_end(self, state: "_State"):
        pass

    def on_loader_start(self, state: "_State"):
        pass

    def on_loader_end(self, state: "_State"):
        pass

    def on_batch_start(self, state: "_State"):
        pass

    def on_batch_end(self, state: "_State"):
        pass

    def on_exception(self, state: "_State"):
        pass


__all__ = [
    "CallbackOrder",
    "CallbackNode",
    "Callback",
]
