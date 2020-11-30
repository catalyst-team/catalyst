# flake8: noqa
from torch import nn
from torch.nn import functional as F


class SimpleNet(nn.Module):
    """Docs? Contribution is welcome"""

    def __init__(
        self,
        num_filters1: int = 6,
        num_filters2: int = 16,
        num_hiddens1: int = 120,
        num_hiddens2: int = 84,
    ):
        """Docs? Contribution is welcome"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_filters1, 5)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.num_hiddens0 = num_filters2 * 5 * 5
        self.fc1 = nn.Linear(self.num_hiddens0, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, 10)

    def forward(self, x):
        """Docs? Contribution is welcome"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_hiddens0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
