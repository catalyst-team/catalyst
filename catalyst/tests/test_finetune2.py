# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, device, engine):
        super().__init__()
        self._logdir = logdir
        self._device = device
        self._engine = engine
        self._name = "finetune2"

    def get_engine(self):
        return self._engine or dl.DeviceEngine(self._device)

    def get_loggers(self):
        loggers = {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }
        if SETTINGS.ml_required:
            loggers["mlflow"]: dl.MLflowLogger(experiment=self._name)
        return loggers

    @property
    def stages(self):
        return ["train_freezed", "train_unfreezed"]

    def get_stage_len(self, stage: str) -> int:
        return 1

    def get_loaders(self, stage: str):
        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32,
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32,
            ),
        }
        return loaders

    def get_model(self, stage: str):
        model = (
            utils.get_nn_from_ddp_module(self.model)
            if self.model is not None
            else nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        )
        if stage == "train_freezed":
            # freeze layer
            utils.set_requires_grad(model[1], False)
        else:
            utils.set_requires_grad(model, True)
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        if stage == "train_freezed":
            return optim.Adam(model.parameters(), lr=1e-3)
        else:
            return optim.SGD(model.parameters(), lr=1e-1)

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str):
        callbacks = {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(
                metric_key="loss",
                grad_clip_fn=nn.utils.clip_grad_norm_,
                grad_clip_params={"max_norm": 1.0},
            ),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="targets", topk_args=(1, 3, 5)
            ),
            "classification": dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        }
        if SETTINGS.ml_required:
            callbacks["confusion_matrix"] = dl.ConfusionMatrixCallback(
                input_key="logits", target_key="targets", num_classes=10
            )
        return callbacks

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


def train_experiment(device, engine=None):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device, engine)
        runner.run()


# Torch
def test_finetune2_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune2_on_torch_cuda0():
    train_experiment("cuda:0")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_finetune2_on_torch_cuda1():
    train_experiment("cuda:1")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found",
)
def test_finetune2_on_torch_dp():
    train_experiment(None, dl.DataParallelEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >=2),
#     reason="No CUDA>=2 found",
# )
# def test_finetune2_on_ddp():
#     train_experiment(None, dl.DistributedDataParallelEngine())

# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found",
)
def test_finetune2_on_amp():
    train_experiment(None, dl.AMPEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_finetune2_on_amp_dp():
    train_experiment(None, dl.DataParallelAMPEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_finetune2_on_amp_ddp():
#     train_experiment(None, dl.DistributedDataParallelAMPEngine())

# APEX
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.apex_required), reason="No CUDA or Apex found",
)
def test_finetune2_on_apex():
    train_experiment(None, dl.APEXEngine())


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
    reason="No CUDA>=2 or Apex found",
)
def test_finetune2_on_apex_dp():
    train_experiment(None, dl.DataParallelApexEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.apex_required),
#     reason="No CUDA>=2 or Apex found",
# )
# def test_finetune2_on_apex_ddp():
#     train_experiment(None, dl.DistributedDataParallelApexEngine())
