# flake8: noqa
import random
import unittest
from unittest.mock import Mock

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.dl import WrapperCallback


class Dummy(Exception):
    pass


def _raise(runner: IRunner):
    raise Dummy()


class RaiserCallback(Callback):
    def __init__(self, order, method_to_raise: str):
        super().__init__(order)
        setattr(self, method_to_raise, _raise)


class TestWrapperCallback(unittest.TestCase):
    def test_loaders_exceptions(self):
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
            wrapper = WrapperCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(
                callback, loaders={"train": ["", "fjdskjfdk", "1234"]}
            )

    def test_ignore_foo_exceptions(self):
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
            wrapper = WrapperCallback(callback, ignore_foo=12345)

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(callback, ignore_foo=lambda arg: True)

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(callback, ignore_foo=lambda *args: True)

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(
                callback, ignore_foo=lambda one, two, three, four: True
            )

        with self.assertRaises(ValueError):
            wrapper = WrapperCallback(
                callback, ignore_foo=lambda *args, **kwargs: True
            )

    def test_ignore_foo_with_wrong_args(self):
        runner = Mock(loader_name="train", epoch=1)
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

        def _ignore_foo(epoch: int, loader: str) -> bool:
            return True

        def _raise_foo(epoch: int, loader: str) -> bool:
            return False

        for order in orders:
            callback = RaiserCallback(order, "on_loader_start")
            wrapper = WrapperCallback(callback, ignore_foo=_ignore_foo)

            wrapper.on_loader_start(runner)

            callback = RaiserCallback(order, "on_loader_start")
            wrapper = WrapperCallback(callback, ignore_foo=_raise_foo)

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
                wrapper = WrapperCallback(callback, ignore_foo=_ignore_foo)

                wrapper.on_loader_start(runner)
                wrapper.__getattribute__(event)(runner)

                callback = RaiserCallback(order, event)
                wrapper = WrapperCallback(callback, ignore_foo=_raise_foo)

                wrapper.on_loader_start(runner)
                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)
