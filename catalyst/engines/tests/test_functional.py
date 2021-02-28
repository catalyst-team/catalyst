# flake8: noqa

import os

from pytest import mark
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from catalyst.engines import (
    DataParallelEngine,
    DeviceEngine,
    DistributedDataParallelEngine,
    process_engine,
)
from catalyst.engines.functional import all_gather, mean_reduce, sum_reduce
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


@mark.skipif(
    not IS_CUDA_AVAILABLE or (NUM_CUDA_DEVICES != 1), reason="CUDA device is not available",
)
def test_device_engine_from_none():
    actual = process_engine(None)
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda"


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_dp_engine_from_none():
    actual = process_engine(None)
    assert isinstance(actual, DataParallelEngine)
    assert actual.device == f"cuda:{torch.cuda.current_device()}"


def test_engine_from_str_on_cpu():
    actual = process_engine("cpu")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cpu"


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_engine_from_str_on_cuda():
    actual = process_engine("cuda")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda"


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_engine_from_str_on_cuda_0():
    actual = process_engine("cuda:0")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda:0"


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_engine_from_str_on_cuda_1():
    actual = process_engine("cuda:1")
    assert isinstance(actual, DeviceEngine)
    assert actual.device == "cuda:1"


def _setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _sum_reduce(rank: int, world_size: int) -> None:
    _setup(rank, world_size)

    to_sreduce = torch.tensor(rank + 1, dtype=torch.float).to(rank)
    actual = sum_reduce(to_sreduce)

    assert actual == torch.tensor((world_size * (world_size + 1)) // 2, dtype=torch.float).to(rank)

    _cleanup()


def _mean_reduce(rank: int, world_size: int) -> None:
    _setup(rank, world_size)

    to_sreduce = torch.tensor(rank + 1, dtype=torch.float).to(rank)
    actual = mean_reduce(to_sreduce, world_size)

    assert actual == torch.tensor(((world_size + 1) / 2), dtype=torch.float).to(rank)

    _cleanup()


def _all_gather(rank, world_size):
    _setup(rank, world_size)

    to_gather = torch.ones(3, dtype=torch.int) * (rank + 1)  # use cpu tensors
    actual = all_gather(to_gather)
    actual = torch.cat(actual)

    expected = torch.cat([torch.ones(3, dtype=torch.int) * (i + 1) for i in range(world_size)])

    assert torch.all(actual.eq(expected))

    _cleanup()


def _run_test(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


@mark.skipif(NUM_CUDA_DEVICES < 2, reason="Need at least 2 CUDA device")
def test_sum_reduce():
    n_gpus = torch.cuda.device_count()
    _run_test(_sum_reduce, n_gpus)


@mark.skipif(NUM_CUDA_DEVICES < 2, reason="Need at least 2 CUDA device")
def test_mean_reduce():
    n_gpus = torch.cuda.device_count()
    _run_test(_mean_reduce, n_gpus)


@mark.skipif(NUM_CUDA_DEVICES < 2, reason="Need at least 2 CUDA device")
def test_all_gather():
    n_gpus = torch.cuda.device_count()
    _run_test(_all_gather, n_gpus)
