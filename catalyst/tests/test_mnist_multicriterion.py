# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl, metrics, utils
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


def train_experiment(device):
    with TemporaryDirectory() as logdir:

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        # <--- multi-criterion setup --->
        criterion = {
            "multiclass": nn.CrossEntropyLoss(),
            "multilabel": nn.BCEWithLogitsLoss(),
        }
        # <--- multi-criterion setup --->

        loaders = {
            "train": DataLoader(
                MNIST(
                    os.getcwd(),
                    train=True,
                    download=True,
                    transform=ToTensor(),
                ),
                batch_size=32,
            ),
            "valid": DataLoader(
                MNIST(
                    os.getcwd(),
                    train=False,
                    download=True,
                    transform=ToTensor(),
                ),
                batch_size=32,
            ),
        }

        class CustomRunner(dl.Runner):
            def predict_batch(self, batch):
                # model inference step
                return self.model(batch[0].to(self.device))

            def on_loader_start(self, runner):
                super().on_loader_start(runner)
                self.meters = {
                    key: metrics.AdditiveValueMetric(compute_on_call=False)
                    for key in ["loss", "accuracy01", "accuracy03"]
                }

            def handle_batch(self, batch):
                # model train/valid step
                # unpack the batch
                x, y = batch
                # run model forward pass
                logits = self.model(x)
                # <--- multi-criterion usage --->
                # compute the loss
                loss_multiclass = self.criterion["multiclass"](logits, y)
                loss_multilabel = self.criterion["multilabel"](
                    logits, F.one_hot(y, 10).to(torch.float32)
                )
                loss = loss_multiclass + loss_multilabel
                # <--- multi-criterion usage --->
                # compute other metrics of interest
                accuracy01, accuracy03 = metrics.accuracy(
                    logits, y, topk=(1, 3)
                )
                # log metrics
                self.batch_metrics.update(
                    {
                        "loss": loss,
                        "accuracy01": accuracy01,
                        "accuracy03": accuracy03,
                    }
                )
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.meters[key].update(
                        self.batch_metrics[key].item(), self.batch_size
                    )
                # run model backward pass
                if self.is_train_loader:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            def on_loader_end(self, runner):
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.loader_metrics[key] = self.meters[key].compute()[0]
                super().on_loader_end(runner)

        runner = CustomRunner()
        # model training
        runner.train(
            engine=dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            num_epochs=1,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
        )


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2,
    reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")
