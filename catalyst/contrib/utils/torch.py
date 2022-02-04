from typing import Callable, Dict, List, Union

import torch
from torch import nn, Tensor
import torch.backends

from catalyst.typing import TorchModel

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


def get_network_output(net: TorchModel, *input_shapes_args, **input_shapes_kwargs):
    """For each input shape returns an output tensor

    Examples:
        >>> net = nn.Linear(10, 5)
        >>> utils.get_network_output(net, (1, 10))
        tensor([[[-0.2665,  0.5792,  0.9757, -0.5782,  0.1530]]])

    Args:
        net: the model
        *input_shapes_args: variable length argument list of shapes
        **input_shapes_kwargs: key-value arguemnts of shapes

    Returns:
        tensor with network output
    """

    def _rand_sample(input_shape) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_shape, dict):
            input_t = {
                key: torch.Tensor(torch.randn((1,) + key_input_shape))
                for key, key_input_shape in input_shape.items()
            }
        else:
            input_t = torch.Tensor(torch.randn((1,) + input_shape))
        return input_t

    input_args = [_rand_sample(input_shape) for input_shape in input_shapes_args]
    input_kwargs = {
        key: _rand_sample(input_shape)
        for key, input_shape in input_shapes_kwargs.items()
    }

    output_t = net(*input_args, **input_kwargs)
    return output_t


def trim_tensors(tensors: Tensor) -> List[torch.Tensor]:
    """
    Trim padding off of a batch of tensors to the smallest possible length.
    Should be used with `catalyst.data.DynamicLenBatchSampler`.

    Adapted from `Dynamic minibatch trimming to improve BERT training speed`_.

    Args:
        tensors: list of tensors to trim.

    Returns:
        List[torch.tensor]: list of trimmed tensors.

    .. _`Dynamic minibatch trimming to improve BERT training speed`:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779
    """
    max_len = torch.max(torch.sum((tensors[0] != 0), 1))
    if max_len > 2:
        tensors = [tsr[:, :max_len] for tsr in tensors]
    return tensors


__all__ = [
    "get_optimal_inner_init",
    "outer_init",
    "get_network_output",
    "trim_tensors",
]
