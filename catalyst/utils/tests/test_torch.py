import torch
from torch import nn

from .. import torch as torch_utils


def test_network_output():
    """
    Test for ``catalyst.utils.torch.get_network_output``
    """
    # case #0
    net = nn.Identity()
    assert torch_utils.get_network_output(net, (1, 20)).shape == (1, 1, 20)

    # case #1
    net = nn.Linear(20, 10)
    assert torch_utils.get_network_output(net, (1, 20)).shape == (1, 1, 10)

    # case #2
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = nn.Linear(20, 10)
            self.net2 = nn.Linear(10, 5)

        def forward(self, x, y):
            z = torch.cat((self.net1(x), self.net2(y)), dim=-1)
            return z

    net = Net()
    assert torch_utils.get_network_output(net, (20,), (10,)).shape == (1, 15)

    # case #3
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(20, 10)

        def forward(self, input):
            z = self.net(input["x"])
            return z

    net = Net()
    input_shapes = {"x": (1, 20)}
    assert torch_utils.get_network_output(net, input_shapes).shape == (1, 1, 10)
