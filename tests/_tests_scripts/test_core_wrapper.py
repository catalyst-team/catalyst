# flake8: noqa
import random
import unittest
from unittest.mock import Mock

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.dl import ControlFlowCallback, WrapperCallback


class Dummy(Exception):
    pass


def _raise(runner: IRunner):
    raise Dummy()


class RaiserCallback(Callback):
    def __init__(self, order, method_to_raise: str):
        super().__init__(order)
        setattr(self, method_to_raise, _raise)


class TestWrapperCallback(unittest.TestCase):
    def test_enabled(self):
        runner = Mock(stage_name="stage1", loader_name="train", epoch=1)

        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
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
                wrapper = WrapperCallback(callback, enable_callback=True)

                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)

    def test_disabled(self):
        runner = Mock(stage_name="stage1", loader_name="train", epoch=1)

        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
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
                wrapper = WrapperCallback(callback, enable_callback=False)
                wrapper.__getattribute__(event)(runner)


class TestControlFlowCallback(unittest.TestCase):
    def test_with_missing_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        for order in orders:
            callback = RaiserCallback(order, "on_epoch_start")
            with self.assertRaises(ValueError):
                ControlFlowCallback(callback)

    def test_epochs_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, epochs=None)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, epochs="123456")

    def test_ignore_epochs_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_epochs=None)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_epochs="123456")

    def test_loaders_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(
                callback, loaders={"train": ["", "fjdskjfdk", "1234"]}
            )

    def test_ignore_loaders_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(
                callback, ignore_loaders={"train": ["", "fjdskjfdk", "1234"]}
            )

    def test_ignore_foo_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, filter_fn=12345)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, filter_fn=lambda arg: True)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, filter_fn=lambda *args: True)

        with self.assertRaises(ValueError):
            ControlFlowCallback(
                callback, filter_fn=lambda one, two, three, four: True
            )

        with self.assertRaises(ValueError):
            ControlFlowCallback(
                callback, filter_fn=lambda *args, **kwargs: True
            )

    def test_filter_fn_with_wrong_args(self):
        runner = Mock(stage_name="stage1", loader_name="train", epoch=1)
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )

        def _ignore_foo(stage: str, epoch: int, loader: str) -> bool:
            return True

        def _raise_foo(stage: str, epoch: int, loader: str) -> bool:
            return False

        for order in orders:
            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(callback, filter_fn=_ignore_foo)

            wrapper.on_loader_start(runner)

            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(callback, filter_fn=_raise_foo)

            with self.assertRaises(Dummy):
                wrapper.on_loader_start(runner)

        events = (
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
                wrapper = ControlFlowCallback(callback, filter_fn=_ignore_foo)

                wrapper.on_loader_start(runner)
                wrapper.__getattribute__(event)(runner)

                callback = RaiserCallback(order, event)
                wrapper = ControlFlowCallback(callback, filter_fn=_raise_foo)

                wrapper.on_loader_start(runner)
                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)

    def test_filter_fn_with_eval(self):
        runner = Mock(stage_name="stage1", loader_name="train", epoch=1)
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
            CallbackOrder.External,
        )

        for order in orders:
            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(
                callback, filter_fn="lambda s, e, l: True"
            )

            wrapper.on_loader_start(runner)

            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(
                callback, filter_fn="lambda s, e, l: False"
            )

            with self.assertRaises(Dummy):
                wrapper.on_loader_start(runner)

        events = (
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
                wrapper = ControlFlowCallback(
                    callback, filter_fn="lambda s, e, l: True"
                )

                wrapper.on_loader_start(runner)
                wrapper.__getattribute__(event)(runner)

                callback = RaiserCallback(order, event)
                wrapper = ControlFlowCallback(
                    callback, filter_fn="lambda s, e, l: False"
                )

                wrapper.on_loader_start(runner)
                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)

    def test_filter_fn_with_err_in_eval(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Validation,
            CallbackOrder.Scheduler,
            CallbackOrder.Logging,
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
                with self.assertRaises(ValueError):
                    ControlFlowCallback(callback, filter_fn="lambda s, e, l")
