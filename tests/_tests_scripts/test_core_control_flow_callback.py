# flake8: noqa
from io import StringIO
import os
import re
import shutil
import sys

import pytest

from catalyst.dl import Callback, CallbackOrder, ControlFlowCallback


class _Runner:
    def __init__(self, stage_name, loader_name, global_epoch, epoch):
        self.stage_name = stage_name
        self.loader_name = loader_name
        self.global_epoch = global_epoch
        self.epoch = epoch


class DummyCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal)


def test_controll_flow_callback_filter_fn_periodical_epochs():
    wraped = ControlFlowCallback(DummyCallback(), epochs=3)
    expected = {
        "train": [i % 3 == 0 for i in range(1, 10 + 1)],
        "valid": [i % 3 == 0 for i in range(1, 10 + 1)],
        "another_loader": [i % 3 == 0 for i in range(1, 10 + 1)],
        "like_valid": [i % 3 == 0 for i in range(1, 10 + 1)],
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
    expected = {
        "train": [i % 4 != 0 for i in range(1, 10 + 1)],
        "valid": [i % 4 != 0 for i in range(1, 10 + 1)],
        "another_loader": [i % 4 != 0 for i in range(1, 10 + 1)],
        "like_valid": [i % 4 != 0 for i in range(1, 10 + 1)],
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
    expected = {
        "train": [
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
        ],
        "valid": [
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
        ],
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
            wraped.on_loader_start(runner)
            actual[loader].append(wraped._is_enabled)
    assert actual == expected


def test_controll_flow_callback_filter_fn_ignore_epochs():
    wraped = ControlFlowCallback(DummyCallback(), ignore_epochs=[3, 4, 6, 8])
    expected = {
        "train": [
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
        ],
        "valid": [
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
        ],
    }
    actual = {loader: [] for loader in expected.keys()}
    for epoch in range(1, 10 + 1):
        for loader in expected.keys():
            runner = _Runner("stage", loader, epoch, epoch)
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
    wraped = ControlFlowCallback(
        DummyCallback(), ignore_loaders=["valid", "another_loader"]
    )
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
    wraped = ControlFlowCallback(
        DummyCallback(), loaders={"valid": 3, "another_loader": [2, 4]}
    )
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
        DummyCallback(),
        filter_fn="lambda stage, epoch, loader: 'valid' in loader",
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
        DummyCallback(),
        filter_fn=lambda stage, epoch, loader: "valid" not in loader,
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
