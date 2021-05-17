# flake8: noqa
import random
import unittest
from unittest.mock import Mock

from catalyst.dl import Callback, CallbackOrder, ControlFlowCallback


class _Runner:
    def __init__(self, stage, loader_key, global_epoch, epoch):
        self.stage_key = stage
        self.loader_key = loader_key
        self.global_epoch_step = global_epoch
        self.stage_epoch_step = epoch


class DummyCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal)


class Dummy(Exception):
    pass


def _raise(runner: "IRunner"):
    raise Dummy()


class RaiserCallback(Callback):
    def __init__(self, order, method_to_raise: str):
        super().__init__(order)
        setattr(self, method_to_raise, _raise)


def test_controll_flow_callback_filter_fn_periodical_epochs():
    wraped = ControlFlowCallback(DummyCallback(), epochs=3)
    mask = [i % 3 == 0 for i in range(1, 10 + 1)]
    expected = {
        "train": mask,
        "valid": mask,
        "another_loader": mask,
        "like_valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_periodical_ignore_epochs():
    wraped = ControlFlowCallback(DummyCallback(), ignore_epochs=4)
    mask = [i % 4 != 0 for i in range(1, 10 + 1)]
    expected = {
        "train": mask,
        "valid": mask,
        "another_loader": mask,
        "like_valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_epochs():
    wraped = ControlFlowCallback(DummyCallback(), epochs=[3, 4, 6])
    mask = [
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    expected = {
        "train": mask,
        "valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_global_epochs():
    wraped = ControlFlowCallback(DummyCallback(), epochs=[3, 4, 7, 10], use_global_epochs=True)
    mask = [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    expected = {
        "train": mask,
        "valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for stage_num, stage in enumerate(["stage1", "stage2"]):
        for epoch in range(1, 5 + 1):
            for loader in expected.keys():
                runner = _Runner(stage, loader, epoch + stage_num * 5, epoch)
                wraped.on_loader_start(runner)
                actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_ignore_epochs():
    wraped = ControlFlowCallback(DummyCallback(), ignore_epochs=[3, 4, 6, 8])
    mask = [
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
    ]
    expected = {
        "train": mask,
        "valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_global_ignore_epochs():
    wraped = ControlFlowCallback(
        DummyCallback(), ignore_epochs=[3, 4, 7, 10], use_global_epochs=True
    )
    mask = [
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    expected = {
        "train": mask,
        "valid": mask,
    }
    actual = {loader: [] for loader in expected.keys()}
    for stage_num, stage in enumerate(["stage1", "stage2"]):
        for epoch in range(1, 5 + 1):
            for loader in expected.keys():
                runner = _Runner(stage, loader, epoch + stage_num * 5, epoch)
                wraped.on_loader_start(runner)
                actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_loaders():
    wraped = ControlFlowCallback(DummyCallback(), loaders=["valid"])
    expected = {
        "train": [False] * 5,
        "valid": [True] * 5,
        "another_loader": [False] * 5,
        "like_valid": [False] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_ignore_loaders():
    wraped = ControlFlowCallback(DummyCallback(), ignore_loaders=["valid", "another_loader"])
    expected = {
        "train": [True] * 5,
        "valid": [False] * 5,
        "another_loader": [False] * 5,
        "like_valid": [True] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_multiple_epochs_loaders():
    wraped = ControlFlowCallback(DummyCallback(), loaders={"valid": 3, "another_loader": [2, 4]})
    expected = {
        "train": [False] * 5,
        "valid": [False, False, True, False, False],
        "another_loader": [False, True, False, True, False],
        "like_valid": [False] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_multiple_epochs_ignore_loaders():
    wraped = ControlFlowCallback(
        DummyCallback(), ignore_loaders={"valid": 3, "another_loader": [2, 4]}
    )
    expected = {
        "train": [True] * 5,
        "valid": [True, True, False, True, True],
        "another_loader": [True, False, True, False, True],
        "like_valid": [True] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_string_lambda():
    wraped = ControlFlowCallback(
        DummyCallback(), filter_fn="lambda stage, epoch, loader: 'valid' in loader",
    )
    expected = {
        "train": [False] * 5,
        "valid": [True] * 5,
        "another_loader": [False] * 5,
        "like_valid": [True] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_control_flow_callback_filter_fn_lambda():
    wraped = ControlFlowCallback(
        DummyCallback(), filter_fn=lambda stage, epoch, loader: "valid" not in loader,
    )
    expected = {
        "train": [True] * 5,
        "valid": [False] * 5,
        "another_loader": [True] * 5,
        "like_valid": [False] * 5,
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 5 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


class TestControlFlowCallback(unittest.TestCase):
    def test_with_missing_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
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
            CallbackOrder.Scheduler,
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
            CallbackOrder.Scheduler,
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
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, loaders={"train": ["", "fjdskjfdk", "1234"]})

    def test_ignore_loaders_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )
        order = random.choice(orders)

        callback = RaiserCallback(order, "on_epoch_start")

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_loaders=1234.56)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, ignore_loaders={"train": ["", "fjdskjfdk", "1234"]})

    def test_ignore_foo_with_wrong_args(self):
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
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
            ControlFlowCallback(callback, filter_fn=lambda one, two, three, four: True)

        with self.assertRaises(ValueError):
            ControlFlowCallback(callback, filter_fn=lambda *args, **kwargs: True)

    def test_filter_fn_with_wrong_args(self):
        runner = Mock(stage="stage1", loader_key="train", epoch=1)
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )

        def _ignore_foo(stage: str, epoch: int, loader: str) -> bool:
            return False

        def _raise_foo(stage: str, epoch: int, loader: str) -> bool:
            return True

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
        runner = Mock(stage="stage1", loader_key="train", epoch=1)
        orders = (
            CallbackOrder.Internal,
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
            CallbackOrder.Optimizer,
            CallbackOrder.Scheduler,
            CallbackOrder.External,
        )

        for order in orders:
            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(callback, filter_fn="lambda s, e, l: False")

            wrapper.on_loader_start(runner)

            callback = RaiserCallback(order, "on_loader_start")
            wrapper = ControlFlowCallback(callback, filter_fn="lambda s, e, l: True")

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
                wrapper = ControlFlowCallback(callback, filter_fn="lambda s, e, l: False")

                wrapper.on_loader_start(runner)
                wrapper.__getattribute__(event)(runner)

                callback = RaiserCallback(order, event)
                wrapper = ControlFlowCallback(callback, filter_fn="lambda s, e, l: True")

                wrapper.on_loader_start(runner)
                with self.assertRaises(Dummy):
                    wrapper.__getattribute__(event)(runner)

    def test_filter_fn_with_err_in_eval(self):
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
                with self.assertRaises(ValueError):
                    ControlFlowCallback(callback, filter_fn="lambda s, e, l")
