# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl, metrics
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

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
    def __init__(self, hid_features, logdir, device, engine):
        super().__init__()
        self.hid_features = hid_features
        self._logdir = logdir
        self._device = device
        self._engine = engine

    def get_engine(self):
        return self._engine or dl.DeviceEngine(self._device)

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 1

    def get_loaders(self, stage: str):
        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
        }
        return loaders

    def get_model(self, stage: str):
        model = self.model if self.model is not None else VAE(28 * 28, self.hid_features)
        return model

    def get_optimizer(self, stage: str, model):
        return optim.Adam(model.parameters(), lr=0.02)

    def get_callbacks(self, stage: str):
        return {
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss_ae", "loss_kld", "loss"]
        }

    def handle_batch(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_, loc, log_scale = self.model(x, deterministic=not self.is_train_loader)

        loss_ae = F.mse_loss(x_, x)
        loss_kld = (-0.5 * torch.sum(1 + log_scale - loc.pow(2) - log_scale.exp(), dim=1)).mean()
        loss = loss_ae + loss_kld * 0.01

        self.batch_metrics = {"loss_ae": loss_ae, "loss_kld": loss_kld, "loss": loss}
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        random_latent_vectors = torch.randn(1, self.hid_features).to(self.device)
        generated_images = self.model.decoder(random_latent_vectors).detach()
        return generated_images


def train_experiment(device, engine=None):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(64, logdir, device, engine)
        runner.run()
        runner.predict_batch(None)[0].cpu().numpy().reshape(28, 28)


# Torch
def test_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_on_torch_cuda0():
    train_experiment("cuda:0")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_on_torch_cuda1():
    train_experiment("cuda:1")


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
# )
# def test_on_torch_dp():
#     train_experiment(None, dl.DataParallelEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >=2),
#     reason="No CUDA>=2 found",
# )
# def test_on_ddp():
#     train_experiment(None, dl.DistributedDataParallelEngine())

# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found",
)
def test_on_amp():
    train_experiment(None, dl.AMPEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_on_amp_dp():
#     train_experiment(None, dl.DataParallelAMPEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_on_amp_ddp():
#     train_experiment(None, dl.DistributedDataParallelAMPEngine())

# APEX
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required), reason="No CUDA or Apex found",
)
def test_on_apex():
    train_experiment(None, dl.APEXEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
#     reason="No CUDA>=2 or Apex found",
# )
# def test_on_apex_dp():
#     train_experiment(None, dl.DataParallelApexEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
#     reason="No CUDA>=2 or Apex found",
# )
# def test_on_apex_ddp():
#     train_experiment(None, dl.DistributedDataParallelApexEngine())
