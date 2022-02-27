# flake8: noqa
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.datasets import MnistMLDataset, MnistQGDataset
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.contrib.models import MnistSimpleNet
from catalyst.data.sampler import BatchBalanceClassSampler
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


class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
            }
        else:
            images, targets, is_query = (
                batch["features"].float(),
                batch["targets"].long(),
                batch["is_query"].bool(),
            )
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "is_query": is_query,
            }


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:

        # 1. train and valid loaders
        train_dataset = MnistMLDataset(root=DATA_ROOT)
        sampler = BatchBalanceClassSampler(
            labels=train_dataset.get_labels(),
            num_classes=5,
            num_samples=10,
            num_batches=10,
        )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=sampler)

        valid_dataset = MnistQGDataset(root=DATA_ROOT, gallery_fraq=0.2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

        # 2. model and optimizer
        model = MnistSimpleNet(out_features=16)
        optimizer = Adam(model.parameters(), lr=0.001)

        # 3. criterion with triplets sampling
        sampler_inbatch = HardTripletsSampler(norm_required=False)
        criterion = TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch
        )

        # 4. training with catalyst Runner
        callbacks = [
            dl.ControlFlowCallbackWrapper(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.ControlFlowCallbackWrapper(
                dl.CMCScoreCallback(
                    embeddings_key="embeddings",
                    labels_key="targets",
                    is_query_key="is_query",
                    topk=[1],
                ),
                loaders="valid",
            ),
            dl.PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="cmc01",
                minimize=False,
                valid=2,
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": valid_loader},
            verbose=False,
            logdir=logdir,
            valid_loader="valid",
            valid_metric="cmc01",
            minimize_valid_metric=False,
            num_epochs=2,
        )


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
