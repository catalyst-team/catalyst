from enum import IntFlag

from .state import State


class CallbackOrder(IntFlag):
    Unknown = -100
    Internal = 0
    Criterion = 20
    Optimizer = 40
    Scheduler = 60
    Metric = 80
    External = 100
    Other = 200


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
    """
    def __init__(self, order: int):
        """
        For order see ``CallbackOrder`` class
        """
        self.order = order

    def on_stage_start(self, state: State):
        pass

    def on_stage_end(self, state: State):
        pass

    def on_epoch_start(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_loader_start(self, state: State):
        pass

    def on_loader_end(self, state: State):
        pass

    def on_batch_start(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass

    def on_exception(self, state: State):
        pass


class LoggerCallback(Callback):
    """
    Loggers are executed on ``start`` before all callbacks,
    and on ``end`` after all callbacks.
    """
    def __init__(self, order: int = None):
        if order is None:
            order = CallbackOrder.Internal
        super().__init__(order=order)
