# flake8: noqa
from torch import nn
from torch.nn import functional as F


class SimpleNet(nn.Module):
    """Docs? Contribution is welcome"""

    def __init__(self, num_hidden1=120, num_hidden2=84):
        """Docs? Contribution is welcome"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 10)

    def forward(self, x):
        """Docs? Contribution is welcome"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
