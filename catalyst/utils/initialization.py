from typing import Callable

import numpy as np
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


def create_optimal_inner_init(
    nonlinearity: nn.Module, **kwargs
) -> Callable[[nn.Module], None]:
    """
    Create initializer for inner layers
    based on their activation function (nonlinearity).
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


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """
    initialize conv/fc bias value according to giving probablity
    """
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
