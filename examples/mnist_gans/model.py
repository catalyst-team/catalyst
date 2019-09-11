import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.contrib import registry


@registry.Model
class SimpleGenerator(nn.Module):
    def __init__(self, noise_dim=10, image_resolution=(28, 28), channels=1):
        super().__init__()
        self.noise_dim = noise_dim
        self.image_resolution = image_resolution
        self.channels = channels
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim, np.prod(image_resolution))
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.size(0), self.channels, *self.image_resolution)
        return x


@registry.Model
class SimpleDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28), channels=1, hid_dim=100):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(channels * np.prod(image_resolution), hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        x = self.net(x.reshape(x.size(0), -1))
        return x
