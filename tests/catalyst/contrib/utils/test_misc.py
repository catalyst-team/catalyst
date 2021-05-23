# flake8: noqa
from torch import nn

from catalyst import utils


def test_get_fn_argsnames():
    class Net1(nn.Module):
        def forward(self, x):
            return x

    class Net2(nn.Module):
        def forward(self, x, y):
            return x

    class Net3(nn.Module):
        def forward(self, x, y=None):
            return x

    class Net4(nn.Module):
        def forward(self, x, *, y=None):
            return x

    class Net5(nn.Module):
        def forward(self, *, x):
            return x

    class Net6(nn.Module):
        def forward(self, *, x, y):
            return x

    class Net7(nn.Module):
        def forward(self, *, x, y=None):
            return x

    nets = [Net1, Net2, Net3, Net4, Net5, Net6, Net7]
    params_true = [
        ["x"],
        ["x", "y"],
        ["x", "y"],
        ["x", "y"],
        ["x"],
        ["x", "y"],
        ["x", "y"],
    ]

    params_predicted = list(
        map(lambda x: utils.get_fn_argsnames(x.forward, exclude=["self"]), nets)
    )
    assert params_predicted == params_true


def test_args_are_not_none():
    """@TODO: Docs. Contribution is welcome."""
    assert utils.args_are_not_none(1, 2, 3, "")
    assert not utils.args_are_not_none(-8, "", None, True)
    assert not utils.args_are_not_none(None)
