# flake8: noqa
import os
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn, optim
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


class CustomRunner(dl.Runner):
    def handle_batch(self, batch):
        x, y1, y2 = batch
        y1_hat, y2_hat = self.model(x)
        self.batch = {
            "features": x,
            "logits1": y1_hat,
            "logits2": y2_hat,
            "targets1": y1,
            "targets2": y2,
        }


class CustomModule(nn.Module):
    def __init__(self, in_features: int, out_features1: int, out_features2: int):
        super().__init__()
        self.shared = nn.Linear(in_features, 128)
        self.head1 = nn.Linear(128, out_features1)
        self.head2 = nn.Linear(128, out_features2)

    def forward(self, x):
        x = self.shared(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        # sample data
        num_samples, num_features, num_classes1, num_classes2 = int(1e4), int(1e1), 4, 10
        X = torch.rand(num_samples, num_features)
        y1 = (torch.rand(num_samples) * num_classes1).to(torch.int64)
        y2 = (torch.rand(num_samples) * num_classes2).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y1, y2)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = CustomModule(num_features, num_classes1, num_classes2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2])

        callbacks = [
            dl.CriterionCallback(
                metric_key="loss1", input_key="logits1", target_key="targets1"
            ),
            dl.CriterionCallback(
                metric_key="loss2", input_key="logits2", target_key="targets2"
            ),
            dl.MetricAggregationCallback(
                metric_key="loss", metrics=["loss1", "loss2"], mode="mean"
            ),
            dl.BackwardCallback(metric_key="loss"),
            dl.OptimizerCallback(metric_key="loss"),
            dl.SchedulerCallback(),
            dl.AccuracyCallback(
                input_key="logits1",
                target_key="targets1",
                num_classes=num_classes1,
                prefix="one_",
            ),
            dl.AccuracyCallback(
                input_key="logits2",
                target_key="targets2",
                num_classes=num_classes2,
                prefix="two_",
            ),
            dl.CheckpointCallback(
                "./logs/one",
                loader_key="valid",
                metric_key="one_accuracy01",
                minimize=False,
                topk=1,
            ),
            dl.CheckpointCallback(
                "./logs/two",
                loader_key="valid",
                metric_key="two_accuracy03",
                minimize=False,
                topk=3,
            ),
        ]
        if SETTINGS.ml_required:
            # catalyst[ml] required
            callbacks.append(
                dl.ConfusionMatrixCallback(
                    input_key="logits1",
                    target_key="targets1",
                    num_classes=num_classes1,
                    prefix="one_cm",
                )
            )
            # catalyst[ml] required
            callbacks.append(
                dl.ConfusionMatrixCallback(
                    input_key="logits2",
                    target_key="targets2",
                    num_classes=num_classes2,
                    prefix="two_cm",
                )
            )

        # model training
        runner = CustomRunner()
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=1,
            verbose=False,
            callbacks=callbacks,
            loggers={
                "console": dl.ConsoleLogger(),
                "tb": dl.TensorboardLogger("./logs/tb"),
            },
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
