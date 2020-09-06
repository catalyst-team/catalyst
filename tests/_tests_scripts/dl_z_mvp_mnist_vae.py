# flake8: noqa
import os

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data.cv import ToTensor
from catalyst.utils import metrics

LOG_SCALE_MAX = 2
LOG_SCALE_MIN = -10


def normal_sample(mu, sigma):
    return mu + sigma * torch.randn_like(sigma)


def normal_logprob(mu, sigma, z):
    normalization_constant = -sigma.log() - 0.5 * np.log(2 * np.pi)
    square_term = -0.5 * ((z - mu) / sigma) ** 2
    logprob_vec = normalization_constant + square_term
    logprob = logprob_vec.sum(1)
    return logprob


class ClassifyVAE(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = torch.nn.Linear(in_features, hid_features * 2)
        self.decoder = nn.Sequential(
            nn.Linear(hid_features, in_features), nn.Sigmoid()
        )
        self.clf = torch.nn.Linear(hid_features, out_features)

    def forward(self, x, deterministic=False):
        z = self.encoder(x)
        bs, z_dim = z.shape

        loc, log_scale = z[:, : z_dim // 2], z[:, z_dim // 2 :]
        log_scale = torch.clamp(log_scale, LOG_SCALE_MIN, LOG_SCALE_MAX)
        scale = torch.exp(log_scale)
        z_ = loc if deterministic else normal_sample(loc, scale)
        z_logprob = normal_logprob(loc, scale, z_)
        z_ = z_.view(bs, -1)
        x_ = self.decoder(z_)
        y_hat = self.clf(z_)

        return y_hat, x_, z_logprob, loc, log_scale


class CustomRunner(dl.Runner):
    """
    Docs.
    """

    def _handle_batch(self, batch):
        """
        Docs.
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_, z_logprob, loc, log_scale = self.model(x)

        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss_kld = (
            -0.5
            * torch.mean(1 + log_scale - loc.pow(2) - log_scale.exp())
            * 0.1
        )
        loss_logprob = torch.mean(z_logprob) * 0.01
        loss = loss_clf + loss_ae + loss_kld + loss_logprob
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(
            y_hat, y, topk=(1, 3, 5)
        )

        self.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss_kld": loss_kld,
            "loss_logprob": loss_logprob,
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
    model = ClassifyVAE(28 * 28, 64, 10)
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
