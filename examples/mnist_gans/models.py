# flake8: noqa
import numpy as np

import torch
import torch.nn as nn

# TODO: add conv models


class SimpleGenerator(nn.Module):
    """Simple fully-connected generator"""
    def __init__(
        self,
        noise_dim=10,
        hidden_dim=256,
        image_resolution=(28, 28),
        channels=1
    ):
        """

        :param noise_dim:
        :param hidden_dim:
        :param image_resolution:
        :param channels:
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, np.prod(image_resolution)), nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.size(0), self.channels, *self.image_resolution)
        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28), channels=1, hidden_dim=100):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(channels * np.prod(image_resolution), hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.net(x.reshape(x.size(0), -1))
        return x


class SimpleCGenerator(SimpleGenerator):
    def __init__(
        self,
        noise_dim=10,
        num_classes=10,
        hidden_dim=256,
        image_resolution=(28, 28),
        channels=1
    ):
        super().__init__(
            noise_dim + num_classes, hidden_dim, image_resolution, channels
        )
        self.num_classes = num_classes

    def forward(self, z, c_one_hot):
        x = torch.cat((z, c_one_hot.float()), dim=1)
        return super().forward(x)


class SimpleCDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes=10,
        image_resolution=(28, 28),
        channels=1,
        hidden_dim=100
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.embedder = nn.Sequential(
            nn.Linear(channels * np.prod(image_resolution), hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05)
        )
        self.classifier = nn.Linear(hidden_dim + num_classes, 1)

    def forward(self, x, c_one_hot):
        x = self.embedder(x.reshape(x.size(0), -1))
        x = self.classifier(torch.cat((x, c_one_hot.float()), dim=1))
        return x


__all__ = [
    "SimpleGenerator", "SimpleDiscriminator", "SimpleCGenerator",
    "SimpleCDiscriminator"
]
