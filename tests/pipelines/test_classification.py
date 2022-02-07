# flake8: noqa
import os
from tempfile import TemporaryDirectory

from pytest import mark

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import (
    IS_CPU_REQUIRED,
    IS_DDP_AMP_REQUIRED,
    IS_DDP_REQUIRED,
    IS_DP_AMP_REQUIRED,
    IS_DP_REQUIRED,
    IS_GPU_AMP_REQUIRED,
    IS_GPU_REQUIRED,
)


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples) * num_classes).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss",
        )
        callbacks = [
            dl.AccuracyCallback(
                input_key="logits", target_key="targets", num_classes=num_classes
            ),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=4
            ),
        ]
        if SETTINGS.ml_required:
            callbacks.append(
                dl.ConfusionMatrixCallback(
                    input_key="logits", target_key="targets", num_classes=4
                )
            )
        if isinstance(engine, dl.CPUEngine):
            callbacks.append(dl.AUCCallback(input_key="logits", target_key="targets"))

        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=1,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=False,
            callbacks=callbacks,
        )


# Device
@mark.skipif(not IS_CPU_REQUIRED, reason="CUDA device is not available")
def test_run_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(
    not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason="CUDA device is not available"
)
def test_run_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


@mark.skipif(
    not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_run_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


# DP
@mark.skipif(
    not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


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


# DDP
@mark.skipif(
    not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_ddp():
    train_experiment(dl.DistributedDataParallelEngine())


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
