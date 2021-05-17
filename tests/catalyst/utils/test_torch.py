# flake8: noqa
from typing import Dict

import torch
from torch import nn

from catalyst.utils import torch as torch_utils


def test_network_output():
    """Test for ``catalyst.utils.torch.get_network_output``."""
    # case #1 - test net with one input variable
    net = nn.Identity()
    assert torch_utils.get_network_output(net, (1, 20)).shape == (1, 1, 20)

    net = nn.Linear(20, 10)
    assert torch_utils.get_network_output(net, (1, 20)).shape == (1, 1, 10)

    # case #2 - test net with several input variables
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = nn.Linear(20, 10)
            self.net2 = nn.Linear(10, 5)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = torch.cat((self.net1(x), self.net2(y)), dim=-1)
            return z

    net = Net()
    assert torch_utils.get_network_output(net, (20,), (10,)).shape == (1, 15)

    # case #3 - test net with key-value input
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(20, 10)

        def forward(self, *, x: torch.Tensor = None) -> torch.Tensor:
            y = self.net(x)
            return y

    net = Net()
    input_shapes = {"x": (20,)}
    assert torch_utils.get_network_output(net, **input_shapes).shape == (1, 10)

    # case #4 - test net with dict of variables input
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(20, 10)

        def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
            y = self.net(x["x"])
            return y

    net = Net()
    input_shapes = {"x": (20,)}
    assert torch_utils.get_network_output(net, input_shapes).shape == (1, 10)
