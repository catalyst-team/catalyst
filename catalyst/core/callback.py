from enum import IntFlag
from typing import Callable, List

from catalyst import utils
from .state import _State


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

    def on_stage_start(self, state: _State):
        pass

    def on_stage_end(self, state: _State):
        pass

    def on_epoch_start(self, state: _State):
        pass

    def on_epoch_end(self, state: _State):
        pass

    def on_loader_start(self, state: _State):
        pass

    def on_loader_end(self, state: _State):
        pass

    def on_batch_start(self, state: _State):
        pass

    def on_batch_end(self, state: _State):
        pass

    def on_exception(self, state: _State):
        pass


class MasterOnlyCallback(Callback):
    pass


class LoggerCallback(MasterOnlyCallback):
    """
    Loggers are executed on ``start`` before all callbacks,
    and on ``end`` after all callbacks.
    """
    def __init__(self, order: int = None):
        if order is None:
            order = CallbackOrder.Internal
        super().__init__(order=order)


class RaiseExceptionCallback(LoggerCallback):
    def __init__(self):
        order = CallbackOrder.Other + 1
        super().__init__(order=order)

    def on_exception(self, state: _State):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if state.need_reraise_exception:
            raise exception


class MetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: _State):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]
        metric = self.metric_fn(outputs, targets, **self.metric_params)
        state.metric_manager.add_batch_value(name=self.prefix, value=metric)


class MultiMetricCallback(Callback):
    """
    A callback that returns multiple metrics on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        list_args: List,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.list_args = list_args
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: _State):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        metrics_ = self.metric_fn(
            outputs, targets, self.list_args, **self.metric_params
        )

        batch_metrics = {}
        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            batch_metrics[key] = metric
        state.metric_manager.add_batch_value(metrics_dict=batch_metrics)


__all__ = [
    "CallbackOrder", "Callable", "LoggerCallback", "MasterOnlyCallback",
    "MetricCallback", "MultiMetricCallback"
]
