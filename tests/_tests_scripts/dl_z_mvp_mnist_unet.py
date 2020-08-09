# flake8: noqa
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data.cv import ToTensor
from catalyst.utils import metrics


class ClassifyUnet(nn.Module):
    def __init__(self, in_channels, in_hw, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.Tanh()
        )
        self.decoder = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.clf = nn.Linear(in_channels * in_hw * in_hw, out_features)

    def forward(self, x):
        z = self.encoder(x)
        z_ = z.view(z.size(0), -1)
        y_hat = self.clf(z_)
        x_ = self.decoder(z)
        return y_hat, x_


class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
        y_hat, x_ = self.model(x_noise)

        loss_clf = F.cross_entropy(y_hat, y)
        iou = metrics.iou(x_, x)
        loss_iou = 1 - iou
        loss = loss_clf + loss_iou
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(
            y_hat, y, topk=(1, 3, 5)
        )

        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_iou": loss_iou,
            "loss": loss,
            "iou": iou,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def main():
    model = ClassifyUnet(1, 28, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(
            MNIST("./data", train=False, download=True, transform=ToTensor(),),
            batch_size=32,
        ),
        "valid": DataLoader(
            MNIST("./data", train=False, download=True, transform=ToTensor(),),
            batch_size=32,
        ),
    }

    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        verbose=True,
        check=True,
    )


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
