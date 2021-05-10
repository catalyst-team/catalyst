# flake8: noqa

from tempfile import TemporaryDirectory

import pytest
import torch
from torch.utils.data import DataLoader

from catalyst.engines import DistributedDataParallelEngine
from catalyst.runners import Runner
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

if SETTINGS.apex_required:
    from catalyst.engines import (
        APEXEngine,
        DataParallelApexEngine,
        DistributedDataParallelApexEngine,
    )

if SETTINGS.amp_required:
    from catalyst.engines import AMPEngine, DataParallelAMPEngine, DistributedDataParallelAMPEngine


class DummyDataset:
    """Dummy dataset."""

    features_dim: int = 4
    out_dim: int = 2

    def __init__(self, num_records: int):
        self.num_records = num_records

    def __len__(self):
        """
        Returns:
            dataset's length.
        """
        return self.num_records

    def __getitem__(self, idx: int):
        """
        Args:
            idx: index of sample

        Returns:
            dummy features and targets vector
        """
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


class EngineIsOk(Exception):
    pass


def get_loaders():
    dataset = DummyDataset(6)
    loader = DataLoader(dataset, batch_size=4)
    return {"train": loader, "valid": loader}


class CustomRunner(Runner):
    _expected_engine = None

    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)
        assert not self._expected_engine is None, "Expected engine is None!"

        assert isinstance(
            self.engine, self._expected_engine
        ), f"Got '{type(self.engine).__name__}' but expected '{self._expected_engine.__name__}'!"

        raise EngineIsOk()

    def handle_batch(self, *args, **kwargs):
        pass


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required and NUM_CUDA_DEVICES == 1),
    reason="AMP is unavailable",
)
def test_amp_arg():
    with TemporaryDirectory(), pytest.raises(EngineIsOk):
        runner = CustomRunner()
        runner._expected_engine = AMPEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), fp16=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required and NUM_CUDA_DEVICES > 1),
    reason="AMP is unavailable or not enough GPUs",
)
def test_dp_amp_arg():
    with TemporaryDirectory(), pytest.raises(EngineIsOk):
        runner = CustomRunner()
        runner._expected_engine = DataParallelAMPEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), fp16=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required and NUM_CUDA_DEVICES == 1),
    reason="APEX is unavailable",
)
def test_apex_arg():
    with TemporaryDirectory(), pytest.raises(EngineIsOk):
        runner = CustomRunner()
        runner._expected_engine = APEXEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), apex=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required and NUM_CUDA_DEVICES > 1),
    reason="APEX is unavailable or not enough GPUs",
)
def test_dp_apex_arg():
    with TemporaryDirectory(), pytest.raises(EngineIsOk):
        runner = CustomRunner()
        runner._expected_engine = DataParallelApexEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), apex=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES > 1), reason="Not enough GPUs",
)
def test_ddp_arg():
    with TemporaryDirectory(), pytest.raises(Exception):
        runner = CustomRunner()
        runner._expected_engine = DistributedDataParallelEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), ddp=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required and NUM_CUDA_DEVICES > 1),
    reason="AMP is unavailable or not enough GPUs",
)
def test_ddp_amp_arg():
    with TemporaryDirectory(), pytest.raises(Exception):
        runner = CustomRunner()
        runner._expected_engine = DistributedDataParallelAMPEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), ddp=True, fp16=True)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required and NUM_CUDA_DEVICES > 1),
    reason="APEX is unavailable or not enough GPUs",
)
def test_ddp_apex_arg():
    with TemporaryDirectory(), pytest.raises(Exception):
        runner = CustomRunner()
        runner._expected_engine = DistributedDataParallelApexEngine
        runner.train(loaders=get_loaders(), model=torch.nn.Linear(4, 2), ddp=True, apex=True)
