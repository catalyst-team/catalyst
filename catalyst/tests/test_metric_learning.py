# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import data, dl
from catalyst.contrib import datasets, models, nn
from catalyst.data.transforms import Compose, Normalize, ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


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


def train_experiment(device, engine=None):
    with TemporaryDirectory() as logdir:

        # 1. train and valid loaders
        transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MnistMLDataset(
            root=os.getcwd(), download=True, transform=transforms
        )
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
        train_loader = DataLoader(
            dataset=train_dataset, sampler=sampler, batch_size=sampler.batch_size
        )

        valid_dataset = datasets.MnistQGDataset(
            root=os.getcwd(), transform=transforms, gallery_fraq=0.2
        )
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

        # 2. model and optimizer
        model = models.MnistSimpleNet(out_features=16)
        optimizer = Adam(model.parameters(), lr=0.001)

        # 3. criterion with triplets sampling
        sampler_inbatch = data.HardTripletsSampler(norm_required=False)
        criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

        # 4. training with catalyst Runner
        callbacks = [
            dl.ControlFlowCallback(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.ControlFlowCallback(
                dl.CMCScoreCallback(
                    embeddings_key="embeddings",
                    labels_key="targets",
                    is_query_key="is_query",
                    topk_args=[1],
                ),
                loaders="valid",
            ),
            dl.PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="cmc01", minimize=False, valid=2
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine or dl.DeviceEngine(device),
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


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_on_torch_dp():
    train_experiment(None, dl.DataParallelEngine())


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


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_on_amp_dp():
    train_experiment(None, dl.DataParallelAMPEngine())


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


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_on_apex_dp():
    train_experiment(None, dl.DataParallelApexEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
#     reason="No CUDA>=2 or Apex found",
# )
# def test_on_apex_ddp():
#     train_experiment(None, dl.DistributedDataParallelApexEngine())
