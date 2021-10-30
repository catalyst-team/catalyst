# flake8: noqa

from distutils.version import LooseVersion
import logging
import os
from tempfile import TemporaryDirectory

import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HAS_REQUIRED_TORCH_VERSION = LooseVersion(torch.__version__) >= LooseVersion("1.8.1")

from catalyst import dl  # noqa: E402
from catalyst.settings import SETTINGS  # noqa: E402

if HAS_REQUIRED_TORCH_VERSION:
    from catalyst.callbacks import ProfilerCallback

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
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


class DummyModel(nn.Module):
    """Docs."""

    def __init__(self, in_features, out_features):
        """Docs."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Linear(in_features, out_features)

    def forward(self, batch):
        """Docs."""
        return self.layers(batch)


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, device, tb_logs=None, chrome_logs=None, stack_logs=None):
        super().__init__()
        self._logdir = logdir
        self._device = device
        self.profiler_tb_logs = tb_logs
        self.chrome_trace_logs = chrome_logs
        self.stacks_logs = stack_logs
        self._export_stacks_kwargs = (
            dict(path=self.stacks_logs) if self.stacks_logs is not None else None
        )

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "profiler": ProfilerCallback(
                loader_key="train",
                epoch=1,
                profiler_kwargs=dict(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=True,
                    with_flops=True,
                ),
                tensorboard_path=self.profiler_tb_logs,
                export_chrome_trace_path=self.chrome_trace_logs,
                export_stacks_kwargs=self._export_stacks_kwargs,
            ),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 10

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        dataset = DummyDataset(6)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return DummyModel(4, 2)

    def get_criterion(self, stage: str):
        return torch.nn.MSELoss()

    def get_optimizer(self, model, stage: str):
        return torch.optim.Adam(model.parameters())

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_trial(self):
        return None

    def get_loggers(self):
        loggers = {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }
        if SETTINGS.mlflow_required:
            loggers["mlflow"] = dl.MLflowLogger(experiment=self.name)

        if SETTINGS.wandb_required:
            loggers["wandb"] = dl.WandbLogger(project="catalyst_test", name=self.name)

        if SETTINGS.neptune_required:
            loggers["neptune"] = dl.NeptuneLogger(
                base_namespace="catalyst-tests",
                api_token="ANONYMOUS",
                project="common/catalyst-integration",
            )

        return loggers

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


def _run_custom_runner(device):
    with TemporaryDirectory() as tmp_dir:
        tb_logs = os.path.join(tmp_dir, "profiler_tb_logs")
        runner = CustomRunner(tmp_dir, device, tb_logs=tb_logs)
        runner.run()
        assert os.path.isdir(tb_logs)

    with TemporaryDirectory() as tmp_dir:
        chrome_logs = os.path.join(tmp_dir, "chrome_trace.json")
        runner = CustomRunner(tmp_dir, device, chrome_logs=chrome_logs)
        runner.run()
        assert os.path.isfile(chrome_logs)

    with TemporaryDirectory() as tmp_dir:
        stack_logs = os.path.join(tmp_dir, "flamegraph.txt")
        runner = CustomRunner(tmp_dir, device, stack_logs=stack_logs)
        runner.run()
        assert os.path.isfile(stack_logs)


@pytest.mark.skipif(
    not HAS_REQUIRED_TORCH_VERSION, reason="Need PyTorch version higher than 1.8.1!"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available!")
def test_profiler_on_cuda():
    _run_custom_runner("cuda:0")
