# flake8: noqa

from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        # sample data
        num_samples, num_features, num_classes1, num_classes2 = (
            int(1e4),
            int(1e1),
            4,
            10,
        )
        X = torch.rand(num_samples, num_features)
        y1 = (torch.rand(num_samples,) * num_classes1).to(torch.int64)
        y2 = (torch.rand(num_samples,) * num_classes2).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y1, y2)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        class CustomModule(nn.Module):
            def __init__(
                self, in_features: int, out_features1: int, out_features2: int
            ):
                super().__init__()
                self.shared = nn.Linear(in_features, 128)
                self.head1 = nn.Linear(128, out_features1)
                self.head2 = nn.Linear(128, out_features2)

            def forward(self, x):
                x = self.shared(x)
                y1 = self.head1(x)
                y2 = self.head2(x)
                return y1, y2

        # model, criterion, optimizer, scheduler
        model = CustomModule(num_features, num_classes1, num_classes2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2])

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

        # model training
        runner = CustomRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=1,
            verbose=False,
            callbacks=[
                dl.CriterionCallback(
                    metric_key="loss1",
                    input_key="logits1",
                    target_key="targets1",
                ),
                dl.CriterionCallback(
                    metric_key="loss2",
                    input_key="logits2",
                    target_key="targets2",
                ),
                dl.MetricAggregationCallback(
                    prefix="loss", metrics=["loss1", "loss2"], mode="mean"
                ),
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
                dl.ConfusionMatrixCallback(
                    input_key="logits1",
                    target_key="targets1",
                    num_classes=num_classes1,
                    prefix="one_cm",
                ),
                # catalyst[ml] required
                dl.ConfusionMatrixCallback(
                    input_key="logits2",
                    target_key="targets2",
                    num_classes=num_classes2,
                    prefix="two_cm",
                ),
                # catalyst[ml] required
                dl.CheckpointCallback(
                    "./logs/one",
                    loader_key="valid",
                    metric_key="one_accuracy",
                    minimize=False,
                    save_n_best=1,
                ),
                dl.CheckpointCallback(
                    "./logs/two",
                    loader_key="valid",
                    metric_key="two_accuracy03",
                    minimize=False,
                    save_n_best=3,
                ),
            ],
            loggers={
                "console": dl.ConsoleLogger(),
                "tb": dl.TensorboardLogger("./logs/tb"),
            },
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
