import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from catalyst import dl


class ClassifyAE(torch.nn.Module):
    """
    Docs.
    """

    def __init__(self, in_features, hid_features, out_features):
        """
        Docs.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hid_features), nn.Tanh()
        )
        self.decoder = nn.Linear(hid_features, in_features)
        self.clf = nn.Linear(hid_features, out_features)

    def forward(self, x):
        """
        Docs.
        """
        z = self.encoder(x)
        y_hat = self.clf(z)
        x_ = self.decoder(z)
        return y_hat, x_


model = ClassifyAE(28 * 28, 128, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(
        MNIST(
            os.getcwd(),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=32,
    ),
    "valid": DataLoader(
        MNIST(
            os.getcwd(),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=32,
    ),
}


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
        y_hat, x_ = self.model(x)
        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss = loss_clf + loss_ae

        self.state.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss": loss,
        }

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()


runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    verbose=True,
    check=True,
)
