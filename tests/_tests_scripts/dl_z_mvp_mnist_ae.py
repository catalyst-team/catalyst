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


class ClassifyAE(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hid_features), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_features, in_features), nn.Sigmoid()
        )
        self.clf = nn.Linear(hid_features, out_features)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.clf(z)
        x_ = self.decoder(z)
        return y_hat, x_


class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_ = self.model(x)
        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss = loss_clf + loss_ae
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(
            y_hat, y, topk=(1, 3, 5)
        )

        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def main():
    model = ClassifyAE(28 * 28, 128, 10)
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
