# flake8: noqa

from pytest import mark

import torch

from catalyst.engines import (
    DataParallelEngine,
    DeviceEngine,
    DistributedDataParallelEngine,
    process_engine,
)
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def test_engine_from_str_on_cpu():
    actual = process_engine("cpu")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cpu"


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_engine_from_str_on_cuda():
    actual = process_engine("cuda")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda:0"


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_engine_from_str_on_cuda_0():
    actual = process_engine("cuda:0")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda:0"


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2,
    reason="Number of CUDA devices is less than 2",
)
def test_engine_from_str_on_cuda_1():
    actual = process_engine("cuda:1")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda:1"


# TODO: add other engine tests
