# flake8: noqa
from pathlib import Path
import subprocess

from pytest import mark

from torch import nn

from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import (
    IS_CONFIGS_REQUIRED,
    IS_CPU_REQUIRED,
    IS_DP_AMP_REQUIRED,
    IS_DP_REQUIRED,
    IS_GPU_AMP_REQUIRED,
    IS_GPU_REQUIRED,
)


class CustomModule(nn.Module):
    def __init__(self, in_features: int, num_hidden: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_hidden),
            nn.Linear(num_hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


def train_experiment_from_configs(*auxiliary_configs: str):
    current_dir = Path(__file__).parent
    main_config = str(current_dir / f"{Path(__file__).stem}.yml")

    engine_configs_dir = current_dir.parent.parent / "pipelines" / "configs"
    auxiliary_configs = " ".join(str(engine_configs_dir / c) for c in auxiliary_configs)

    script = Path("catalyst", "contrib", "scripts", "tune.py")
    cmd = f"python {script} -C {main_config} {auxiliary_configs} --n-trials 2"
    subprocess.run(cmd.split(), check=True)


# Device
@mark.skipif(
    not SETTINGS.optuna_required or not IS_CONFIGS_REQUIRED or not IS_CPU_REQUIRED,
    reason="CPU device is not available",
)
def test_config_run_on_cpu():
    train_experiment_from_configs("engine_cpu.yml")


@mark.skipif(
    not SETTINGS.optuna_required
    or not IS_CONFIGS_REQUIRED
    or not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]),
    reason="CUDA device is not available",
)
def test_config_run_on_torch_cuda0():
    train_experiment_from_configs("engine_gpu.yml")


@mark.skipif(
    not SETTINGS.optuna_required
    or not IS_CONFIGS_REQUIRED
    or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_config_run_on_amp():
    train_experiment_from_configs("engine_gpu_amp.yml")


@mark.skipif(
    not SETTINGS.optuna_required
    or not IS_CONFIGS_REQUIRED
    or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_config_run_on_torch_dp():
    train_experiment_from_configs("engine_dp.yml")


@mark.skipif(
    not SETTINGS.optuna_required
    or not IS_CONFIGS_REQUIRED
    or not all(
        [
            IS_DP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_config_run_on_amp_dp():
    train_experiment_from_configs("engine_dp_amp.yml")
