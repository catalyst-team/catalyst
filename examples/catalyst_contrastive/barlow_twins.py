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


class CifarPairTransform:
    def __init__(self, train_transform=True, pair_transform=True):
        if train_transform is True:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)


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


class CustomRunner(dl.Runner):
    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            (pos_1, pos_2), targets = batch
            feature_1, out_1 = self.model(pos_1)
            _, out_2 = self.model(pos_2)
            self.batch = {
                "embeddings": feature_1,
                "out_1": out_1,
                "out_2": out_2,
                "targets": targets,
            }
        else:
            images, targets = batch
            features, _ = self.model(images)
            self.batch = {"embeddings": features, "targets": targets}


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
    "--logdir", default="./logdir", type=str, help="Logs directory (tensorboard, weights, etc)",
)
if __name__ == "__main__":

    # args parse
    args = parser.parse_args()

    # hyperparams
    feature_dim, temperature = args.feature_dim, args.temperature
    offdig_lambda = args.offdig_lambda
    batch_size, epochs, num_workers = args.batch_size, args.epochs, args.num_workers

    # data
    train_data = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=CifarPairTransform(train_transform=True), download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        transform=CifarPairTransform(train_transform=False, pair_transform=False),
        download=True,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(input_key="out_1", target_key="out_2", metric_key="loss"),
            loaders="train",
        ),
        dl.SklearnModelCallback(
            feature_key="embeddings",
            target_key="targets",
            train_loader="train",
            valid_loaders="valid",
            model_fn=LogisticRegression,
            predict_key="sklearn_predict",
            predict_method="predict_proba",
        ),
        dl.OptimizerCallback(metric_key="loss"),
        dl.ControlFlowCallback(
            dl.AccuracyCallback(
                target_key="targets", input_key="sklearn_predict", topk_args=(1, 3)
            ),
            loaders="valid",
        ),
    ]

    model = Model(feature_dim, arch="resnet50")
    criterion = BarlowTwinsLoss(offdiag_lambda=offdig_lambda)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

    runner = CustomRunner()

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
