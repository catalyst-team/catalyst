# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        student = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = {"cls": nn.CrossEntropyLoss(), "kl": nn.KLDivLoss(reduction="batchmean")}
        optimizer = optim.Adam(student.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
        }

        class DistilRunner(dl.Runner):
            def handle_batch(self, batch):
                x, y = batch

                teacher.eval()  # let's manually set teacher model to eval mode
                with torch.no_grad():
                    t_logits = self.model["teacher"](x)

                s_logits = self.model["student"](x)
                self.batch = {
                    "t_logits": t_logits,
                    "s_logits": s_logits,
                    "targets": y,
                    "s_logprobs": F.log_softmax(s_logits, dim=-1),
                    "t_probs": F.softmax(t_logits, dim=-1),
                }

        runner = DistilRunner()
        # model training
        runner.train(
            engine=dl.DeviceEngine(device),
            model={"teacher": teacher, "student": student},
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            logdir=logdir,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="t_logits", target_key="targets", num_classes=2, prefix="teacher_"
                ),
                dl.AccuracyCallback(
                    input_key="s_logits", target_key="targets", num_classes=2, prefix="student_"
                ),
                dl.CriterionCallback(
                    input_key="s_logits",
                    target_key="targets",
                    metric_key="cls_loss",
                    criterion_key="cls",
                ),
                dl.CriterionCallback(
                    input_key="s_logprobs",
                    target_key="t_probs",
                    metric_key="kl_div_loss",
                    criterion_key="kl",
                ),
                dl.MetricAggregationCallback(
                    prefix="loss", metrics=["kl_div_loss", "cls_loss"], mode="mean"
                ),
                dl.OptimizerCallback(metric_key="loss", model_key="student"),
                dl.CheckpointCallback(
                    logdir=logdir,
                    loader_key="valid",
                    metric_key="loss",
                    minimize=True,
                    save_n_best=3,
                ),
            ],
        )


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")
