from typing import Callable, List, TYPE_CHECKING  # isort:skip
from enum import IntFlag
from functools import partial

from catalyst import utils
if TYPE_CHECKING:
    from .state import _State


class CallbackOrder(IntFlag):
    Internal = 0
    Metric = 20
    MetricAggregation = 40
    Optimizer = 60
    Scheduler = 80
    Logging = 100
    External = 120
    Other = 200


class CallbackNode(IntFlag):
    All = 0
    Master = 1
    Worker = 2


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
    def __init__(self, order: int, node: int = CallbackNode.All):
        """
        For order see ``CallbackOrder`` class
        """
        self.order = order
        self.node = node

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


class RaiseExceptionCallback(Callback):
    def __init__(self):
        order = CallbackOrder.Other + 1
        super().__init__(order=order, node=CallbackNode.All)

    def on_exception(self, state: "_State"):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if state.need_exception_reraise:
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
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = partial(metric_fn, **metric_params)
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: "_State"):
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]
        metric = self.metric_fn(outputs, targets)
        state.batch_metrics[self.prefix] = metric


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
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = partial(metric_fn, **metric_params)
        self.list_args = list_args
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: "_State"):
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]

        metrics_ = self.metric_fn(outputs, targets, self.list_args)

        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            state.batch_metrics[key] = metric


__all__ = [
    "CallbackOrder", "CallbackNode", "Callback",
    "MetricCallback", "MultiMetricCallback"
]
