from typing import Callable  # isort:skip

import torch.nn as nn

ACTIVATIONS = {
    None: "sigmoid",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.ELU: "relu",
}


def _nonlinearity2name(nonlinearity):
    if isinstance(nonlinearity, nn.Module):
        nonlinearity = nonlinearity.__class__
    nonlinearity = ACTIVATIONS.get(nonlinearity, nonlinearity)
    nonlinearity = nonlinearity.lower()
    return nonlinearity


def get_optimal_inner_init(
    nonlinearity: nn.Module, **kwargs
) -> Callable[[nn.Module], None]:
    """
    Create initializer for inner layers
    based on their activation function (nonlinearity).

    Args:
        nonlinearity: non-linear activation
    """
    nonlinearity: str = _nonlinearity2name(nonlinearity)
    assert isinstance(nonlinearity, str)

    if nonlinearity in ["sigmoid", "tanh"]:
        weignt_init_fn = nn.init.xavier_uniform_
        init_args = kwargs
    elif nonlinearity in ["relu", "leaky_relu"]:
        weignt_init_fn = nn.init.kaiming_normal_
        init_args = {**{"nonlinearity": nonlinearity}, **kwargs}
    else:
        raise NotImplementedError

    def inner_init(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weignt_init_fn(layer.weight.data, **init_args)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)

    return inner_init


def outer_init(layer: nn.Module) -> None:
    """
    Initialization for output layers of policy and value networks typically
    used in deep reinforcement learning literature.
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)
