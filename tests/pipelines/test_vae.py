# flake8: noqa
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl, metrics
from catalyst.contrib.datasets import MNIST
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import (
    DATA_ROOT,
    IS_CONFIGS_REQUIRED,
    IS_CPU_REQUIRED,
    IS_DDP_AMP_REQUIRED,
    IS_DDP_REQUIRED,
    IS_DP_AMP_REQUIRED,
    IS_DP_REQUIRED,
    IS_GPU_AMP_REQUIRED,
    IS_GPU_REQUIRED,
)
from tests.misc import run_experiment_from_configs

LOG_SCALE_MAX = 2
LOG_SCALE_MIN = -10


def normal_sample(loc, log_scale):
    scale = torch.exp(0.5 * log_scale)
    return loc + scale * torch.randn_like(scale)


class VAE(nn.Module):
    def __init__(self, in_features, hid_features):
        super().__init__()
        self.hid_features = hid_features
        self.encoder = nn.Linear(in_features, hid_features * 2)
        self.decoder = nn.Sequential(nn.Linear(hid_features, in_features), nn.Sigmoid())

    def forward(self, x, deterministic=False):
        z = self.encoder(x)
        bs, z_dim = z.shape

        loc, log_scale = z[:, : z_dim // 2], z[:, z_dim // 2 :]
        log_scale = torch.clamp(log_scale, LOG_SCALE_MIN, LOG_SCALE_MAX)

        z_ = loc if deterministic else normal_sample(loc, log_scale)
        z_ = z_.view(bs, -1)
        x_ = self.decoder(z_)

        return x_, loc, log_scale


class CustomRunner(dl.IRunner):
    def __init__(self, hid_features, logdir, engine):
        super().__init__()
        self.hid_features = hid_features
        self._logdir = logdir
        self._engine = engine

    def get_engine(self):
        return self._engine

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def num_epochs(self) -> int:
        return 1

    def get_loaders(self):
        loaders = {
            "train": DataLoader(
                MNIST(DATA_ROOT, train=False),
                batch_size=32,
            ),
            "valid": DataLoader(
                MNIST(DATA_ROOT, train=False),
                batch_size=32,
            ),
        }
        return loaders

    def get_model(self):
        model = self.model if self.model is not None else VAE(28 * 28, self.hid_features)
        return model

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0.02)

    def get_callbacks(self):
        return {
            "backward": dl.BackwardCallback(metric_key="loss"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                topk=3,
            ),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss_ae", "loss_kld", "loss"]
        }

    def handle_batch(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_, loc, log_scale = self.model(x, deterministic=not self.is_train_loader)

        loss_ae = F.mse_loss(x_, x)
        loss_kld = (
            -0.5 * torch.sum(1 + log_scale - loc.pow(2) - log_scale.exp(), dim=1)
        ).mean()
        loss = loss_ae + loss_kld * 0.01

        self.batch_metrics = {"loss_ae": loss_ae, "loss_kld": loss_kld, "loss": loss}
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        random_latent_vectors = torch.randn(1, self.hid_features).to(self.engine.device)
        generated_images = self.model.decoder(random_latent_vectors).detach()
        return generated_images


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(64, logdir, engine)
        runner.run()
        if isinstance(engine, (dl.CPUEngine, dl.GPUEngine)):
            runner.predict_batch(None)[0].cpu().numpy().reshape(28, 28)


def train_experiment_from_configs(*auxiliary_configs: str):
    run_experiment_from_configs(
        Path(__file__).parent / "configs",
        f"{Path(__file__).stem}.yml",
        *auxiliary_configs,
    )


# Device
@mark.skipif(not IS_CPU_REQUIRED, reason="CUDA device is not available")
def test_run_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED or not IS_CPU_REQUIRED, reason="CPU device is not available"
)
def test_config_run_on_cpu():
    train_experiment_from_configs("engine_cpu.yml")


@mark.skipif(
    not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason="CUDA device is not available"
)
def test_run_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED or not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]),
    reason="CUDA device is not available",
)
def test_config_run_on_torch_cuda0():
    train_experiment_from_configs("engine_gpu.yml")


@mark.skipif(
    not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_run_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_config_run_on_amp():
    train_experiment_from_configs("engine_gpu_amp.yml")


# DP
@mark.skipif(
    not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_config_run_on_torch_dp():
    train_experiment_from_configs("engine_dp.yml")


@mark.skipif(
    not all(
        [
            IS_DP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_run_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


@mark.skipif(
    not IS_CONFIGS_REQUIRED
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


# DDP
# @mark.skipif(
#     not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# @mark.skipif(
#     not IS_CONFIGS_REQUIRED
#     or not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_config_run_on_torch_ddp():
#     train_experiment_from_configs("engine_ddp.yml")


# @mark.skipif(
#     not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))


# @mark.skipif(
#     not IS_CONFIGS_REQUIRED
#     or not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_config_run_on_amp_ddp():
#     train_experiment_from_configs("engine_ddp_amp.yml")


# def _train_fn(local_rank, world_size):
#     process_group_kwargs = {
#         "backend": "nccl",
#         "world_size": world_size,
#     }
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     dist.init_process_group(**process_group_kwargs)
#     train_experiment(dl.Engine())
#     dist.destroy_process_group()


# @mark.skipif(
#     not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp_spawn():
#     world_size: int = torch.cuda.device_count()
#     mp.spawn(
#         _train_fn,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


# def _train_fn_amp(local_rank, world_size):
#     process_group_kwargs = {
#         "backend": "nccl",
#         "world_size": world_size,
#     }
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     dist.init_process_group(**process_group_kwargs)
#     train_experiment(dl.Engine(fp16=True))
#     dist.destroy_process_group()


# @mark.skipif(
#     not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_torch_ddp_amp_spawn():
#     world_size: int = torch.cuda.device_count()
#     mp.spawn(
#         _train_fn_amp,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )
#     dist.destroy_process_group()
