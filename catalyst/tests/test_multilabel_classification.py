# flake8: noqa

from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


def train_experiment(device, engine=None):
    with TemporaryDirectory() as logdir:
        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        runner.train(
            engine=engine or dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=1,
            valid_loader="valid",
            valid_metric="accuracy",
            minimize_valid_metric=False,
            verbose=False,
            callbacks=[
                dl.AUCCallback(input_key="logits", target_key="targets"),
                dl.MultilabelAccuracyCallback(
                    input_key="logits", target_key="targets", threshold=0.5
                ),
            ],
        )


# Torch
def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DataParallelEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.AMPEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DataParallelAMPEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DistributedDataParallelAMPEngine())


# APEX
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required), reason="No CUDA or Apex found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.APEXEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DataParallelApexEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_finetune_on_cuda_device():
    train_experiment(None, dl.DistributedDataParallelApexEngine())
