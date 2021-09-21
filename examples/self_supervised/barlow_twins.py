# flake8: noqa
import argparse

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
        self.encoder = nn.Sequential(ResnetEncoder(**resnet_kwargs), nn.Flatten())
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
parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
parser.add_argument("--temperature", default=0.5, type=float, help="Temperature used in softmax")
parser.add_argument(
    "--batch_size", default=512, type=int, help="Number of images in each mini-batch"
)
parser.add_argument(
    "--epochs", default=1000, type=int, help="Number of sweeps over the dataset to train"
)
parser.add_argument(
    "--num_workers", default=8, type=float, help="Number of workers to process a dataloader"
)
parser.add_argument(
    "--offdig_lambda",
    default=0.005,
    type=float,
    help="Lambda that controls the on- and off-diagonal terms",
)
parser.add_argument(
    "--logdir", default="./logdir", type=str, help="Logs directory (tensorboard, weights, etc)"
)
if __name__ == "__main__":

    # args parse
    args = parser.parse_args()

    # hyperparams
    feature_dim, temperature = args.feature_dim, args.temperature
    offdig_lambda = args.offdig_lambda
    batch_size, epochs, num_workers = args.batch_size, args.epochs, args.num_workers

    # data

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ]
    )

    transform_original = transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    train_data = SelfSupervisedDatasetWrapper(
        torchvision.datasets.CIFAR10(root="data", train=True, transform=None, download=True),
        transforms=transforms,
        transform_original=transform_original,
    )
    test_data = SelfSupervisedDatasetWrapper(
        torchvision.datasets.CIFAR10(root="data", train=False, transform=None, download=True),
        transforms=transforms,
        transform_original=transform_original,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(
                input_key="projection_left", target_key="projection_right", metric_key="loss"
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
        # dl.OptimizerCallback(metric_key="loss"),
        # dl.ControlFlowCallback(
        #     dl.AccuracyCallback(
        #         target_key="target", input_key="sklearn_predict", topk_args=(1, 3)
        #     ),
        #     loaders="valid",
        # ),
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
        overfit=True,
    )
