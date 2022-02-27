# flake8: noqa
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.layers import Lambda
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


class CustomRunner(dl.Runner):
    def __init__(self, *args, latent_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim

    def predict_batch(self, batch):
        batch_size = 1
        # Sample random points in the latent space
        random_latent_vectors = torch.randn(batch_size, self.latent_dim).to(
            self.engine.device
        )
        # Decode them to fake images
        generated_images = self.model["generator"](random_latent_vectors).detach()
        return generated_images

    def handle_batch(self, batch):
        real_images, _ = batch
        batch_size = real_images.shape[0]

        # Sample random points in the latent space
        random_latent_vectors = torch.randn(batch_size, self.latent_dim).to(
            self.engine.device
        )

        # Decode them to fake images
        generated_images = self.model["generator"](random_latent_vectors).detach()
        # Combine them with real images
        combined_images = torch.cat([generated_images, real_images])

        # Assemble labels discriminating real from fake images
        labels = torch.cat(
            [torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))]
        ).to(self.engine.device)
        # Add random noise to the labels - important trick!
        labels += 0.05 * torch.rand(labels.shape).to(self.engine.device)

        # Discriminator forward
        combined_predictions = self.model["discriminator"](combined_images)

        # Sample random points in the latent space
        random_latent_vectors = torch.randn(batch_size, self.latent_dim).to(
            self.engine.device
        )
        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((batch_size, 1)).to(self.engine.device)

        # Generator forward
        generated_images = self.model["generator"](random_latent_vectors)
        generated_predictions = self.model["discriminator"](generated_images)

        self.batch = {
            "combined_predictions": combined_predictions,
            "labels": labels,
            "generated_predictions": generated_predictions,
            "misleading_labels": misleading_labels,
        }


def _ddp_hack(x):
    return x.view(x.size(0), 1, 28, 28)


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        # latent_dim = 128
        # generator = nn.Sequential(
        #     # We want to generate 128 coefficients to reshape into a 7x7x128 map
        #     nn.Linear(128, 128 * 7 * 7),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     Lambda(lambda x: x.view(x.size(0), 128, 7, 7)),
        #     nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 1, (7, 7), padding=3),
        #     nn.Sigmoid(),
        # )
        # discriminator = nn.Sequential(
        #     nn.Conv2d(1, 64, (3, 3), stride=(2, 2), padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     GlobalMaxPool2d(),
        #     Flatten(),
        #     nn.Linear(128, 1),
        # )
        latent_dim = 32
        generator = nn.Sequential(
            nn.Linear(latent_dim, 28 * 28), Lambda(_ddp_hack), nn.Sigmoid()
        )
        discriminator = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 1))

        model = {"generator": generator, "discriminator": discriminator}
        criterion = {
            "generator": nn.BCEWithLogitsLoss(),
            "discriminator": nn.BCEWithLogitsLoss(),
        }
        optimizer = {
            "generator": torch.optim.Adam(
                generator.parameters(), lr=0.0003, betas=(0.5, 0.999)
            ),
            "discriminator": torch.optim.Adam(
                discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)
            ),
        }
        loaders = {
            "train": DataLoader(
                MNIST(DATA_ROOT, train=False),
                batch_size=32,
            ),
        }

        runner = CustomRunner(latent_dim=latent_dim)
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            callbacks=[
                dl.CriterionCallback(
                    input_key="combined_predictions",
                    target_key="labels",
                    metric_key="loss_discriminator",
                    criterion_key="discriminator",
                ),
                dl.BackwardCallback(metric_key="loss_discriminator"),
                dl.OptimizerCallback(
                    optimizer_key="discriminator",
                    metric_key="loss_discriminator",
                ),
                dl.CriterionCallback(
                    input_key="generated_predictions",
                    target_key="misleading_labels",
                    metric_key="loss_generator",
                    criterion_key="generator",
                ),
                dl.BackwardCallback(metric_key="loss_generator"),
                dl.OptimizerCallback(
                    optimizer_key="generator",
                    metric_key="loss_generator",
                ),
            ],
            valid_loader="train",
            valid_metric="loss_generator",
            minimize_valid_metric=True,
            num_epochs=1,
            verbose=False,
            logdir=logdir,
        )
        if isinstance(engine, (dl.CPUEngine, dl.GPUEngine)) and not engine.is_ddp:
            runner.predict_batch(None)[0, 0].cpu().numpy()


def train_experiment_from_configs(*auxiliary_configs: str):
    run_experiment_from_configs(
        Path(__file__).parent / "configs",
        f"{Path(__file__).stem}.yml",
        *auxiliary_configs,
    )


# Device
@mark.skipif(not IS_CPU_REQUIRED, reason="CPU device is not available")
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
@mark.skipif(
    not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_ddp():
    train_experiment(dl.DistributedDataParallelEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_config_run_on_torch_ddp():
    train_experiment_from_configs("engine_ddp.yml")


@mark.skipif(
    not all(
        [
            IS_DDP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_run_on_amp_ddp():
    train_experiment(dl.DistributedDataParallelEngine(fp16=True))


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all(
        [
            IS_DDP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_config_run_on_amp_ddp():
    train_experiment_from_configs("engine_ddp_amp.yml")


def _train_fn(local_rank, world_size):
    process_group_kwargs = {
        "backend": "nccl",
        "world_size": world_size,
    }
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    dist.init_process_group(**process_group_kwargs)
    train_experiment(dl.Engine())
    dist.destroy_process_group()


@mark.skipif(
    not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_ddp_spawn():
    world_size: int = torch.cuda.device_count()
    mp.spawn(
        _train_fn,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


def _train_fn_amp(local_rank, world_size):
    process_group_kwargs = {
        "backend": "nccl",
        "world_size": world_size,
    }
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    dist.init_process_group(**process_group_kwargs)
    train_experiment(dl.Engine(fp16=True))
    dist.destroy_process_group()


@mark.skipif(
    not all(
        [
            IS_DDP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_run_on_torch_ddp_amp_spawn():
    world_size: int = torch.cuda.device_count()
    mp.spawn(
        _train_fn_amp,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
