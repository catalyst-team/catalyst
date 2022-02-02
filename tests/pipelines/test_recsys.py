# flake8: noqa

from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        callbacks = [
            dl.BatchTransformCallback(
                input_key="logits",
                output_key="scores",
                transform=torch.sigmoid,
                scope="on_batch_end",
            ),
            dl.CriterionCallback(
                input_key="logits", target_key="targets", metric_key="loss"
            ),
            dl.AUCCallback(input_key="scores", target_key="targets"),
            dl.HitrateCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
            dl.MRRCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
            dl.MAPCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
            dl.NDCGCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
            dl.OptimizerCallback(metric_key="loss"),
            dl.SchedulerCallback(),
            dl.CheckpointCallback(
                logdir=logdir, loader_key="valid", metric_key="map01", minimize=False
            ),
        ]
        if isinstance(engine, dl.CPUEngine):
            callbacks.append(dl.AUCCallback(input_key="logits", target_key="targets"))

        # model training
        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss",
        )
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
        )


# Torch
def test_classification_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_classification_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_cuda1():
#     train_experiment("cuda:1")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
)
def test_classification_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
)
def test_classification_on_torch_ddp():
    train_experiment(dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found"
)
def test_classification_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_ddp():
    train_experiment(dl.DistributedDataParallelEngine(fp16=True))
