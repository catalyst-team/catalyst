# flake8: noqa
import random
import unittest
from unittest.mock import Mock

from catalyst.core.callback import Callback, CallbackOrder, CallbackWrapper


class Dummy(Exception):
    pass


def _raise(runner: "IRunner"):
    raise Dummy()


class RaiserCallback(Callback):
    def __init__(self, order, method_to_raise: str):
        super().__init__(order)
        setattr(self, method_to_raise, _raise)


class TestWrapperCallback(unittest.TestCase):
    def test_enabled(self):
        runner = Mock(stage="stage1", loader_key="train", epoch=1)

        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )

        events = (
            "on_loader_start",
            "on_loader_end",
            "on_stage_start",
            "on_stage_end",
            "on_epoch_start",
            "on_epoch_end",
            "on_batch_start",
            "on_batch_end",
            "on_exception",
        )

        for event in events:
            for order in orders:
                callback = RaiserCallback(order, event)
                wrapper = CallbackWrapper(callback, enable_callback=True)

                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)

    def test_disabled(self):
        runner = Mock(stage="stage1", loader_key="train", epoch=1)

        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )

        events = (
            "on_loader_start",
            "on_loader_end",
            "on_stage_start",
            "on_stage_end",
            "on_epoch_start",
            "on_epoch_end",
            "on_batch_start",
            "on_batch_end",
            "on_exception",
        )

        for event in events:
            for order in orders:
                callback = RaiserCallback(order, event)
                wrapper = CallbackWrapper(callback, enable_callback=False)
                wrapper.__getattribute__(event)(runner)
