# flake8: noqa
import argparse

from common import add_arguments, datasets
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from catalyst import dl
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn import BarlowTwinsLoss
from catalyst.data import SelfSupervisedDatasetWrapper


class Model(nn.Module):
    def __init__(self, feature_dim=128, **resnet_kwargs):
        super(Model, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            ResnetEncoder(**resnet_kwargs), nn.Flatten()
        )
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        feature = self.encoder(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


parser = argparse.ArgumentParser(description="Train Barlow Twins on cifar-10")
add_arguments(parser)
parser.add_argument(
    "--offdig_lambda",
    default=0.005,
    type=float,
    help="Lambda that controls the on- and off-diagonal terms",
)

if __name__ == "__main__":

    # args parse
    args = parser.parse_args()

    # hyperparams
    feature_dim, temperature = args.feature_dim, args.temperature
    offdig_lambda = args.offdig_lambda
    batch_size, epochs, num_workers = (
        args.batch_size,
        args.epochs,
        args.num_workers,
    )
    dataset = args.dataset
    # data

    transforms = datasets[dataset]["train_transform"]
    transform_original = datasets[dataset]["valid_transform"]

    train_data = SelfSupervisedDatasetWrapper(
        datasets[dataset]["dataset"](
            root="data", train=True, transform=None, download=True
        ),
        transforms=transforms,
        transform_original=transform_original,
    )
    test_data = SelfSupervisedDatasetWrapper(
        datasets[dataset]["dataset"](
            root="data", train=False, transform=None, download=True
        ),
        transforms=transforms,
        transform_original=transform_original,
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=True
    )

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(
                input_key="projection_left",
                target_key="projection_right",
                metric_key="loss",
            ),
            loaders="train",
        ),
        dl.SklearnModelCallback(
            feature_key="embedding_origin",
            target_key="target",
            train_loader="train",
            valid_loaders="valid",
            model_fn=LogisticRegression,
            predict_key="sklearn_predict",
            predict_method="predict_proba",
        ),
        dl.OptimizerCallback(metric_key="loss"),
        dl.ControlFlowCallback(
            dl.AccuracyCallback(
                target_key="target",
                input_key="sklearn_predict",
                topk_args=(1, 3),
            ),
            loaders="valid",
        ),
    ]

    model = Model(feature_dim, arch="resnet50")
    criterion = BarlowTwinsLoss(offdiag_lambda=offdig_lambda)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

    runner = dl.SelfSupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": valid_loader},
        verbose=True,
        num_epochs=epochs,
        valid_loader="train",
        valid_metric="loss",
        logdir=args.logdir,
        minimize_valid_metric=True,
    )
