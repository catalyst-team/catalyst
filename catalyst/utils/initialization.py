from typing import Callable

from torch import nn

# TODO: move to global registry with activation functions
ACTIVATIONS = {  # noqa: WPS407
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
        **kwargs: extra kwargs

    Returns:
        optimal initialization function

    Raises:
        NotImplementedError: if nonlinearity is out of
            `sigmoid`, `tanh`, `relu, `leaky_relu`
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

    Args:
        layer: torch nn.Module instance
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)


def reset_weights_if_possible(module: nn.Module):
    """
    Resets module parameters if possible.

    Args:
        module: Module to reset.
    """
    try:
        module.reset_parameters()
    except AttributeError:
        pass


__all__ = ["get_optimal_inner_init", "outer_init", "reset_weights_if_possible"]
