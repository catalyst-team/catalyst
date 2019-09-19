from typing import List, Union
from functools import partial

import torch.nn as nn

from catalyst.contrib.registry import MODULES
from catalyst import utils
from .sequential import SequentialNet


def get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    kernel_sizes: List = None,
    strides: List = None,
    groups: List = None,
    use_bias: bool = False,
    normalization: str = None,
    dropout_rate: float = None,
    activation: str = "ReLU"
) -> nn.Module:

    channels = channels or [32, 64, 64]
    kernel_sizes = kernel_sizes or [8, 4, 3]
    strides = strides or [4, 2, 1]
    groups = groups or [1, 1, 1]
    activation_fn = nn.__dict__[activation]
    assert len(channels) == len(kernel_sizes) == len(strides) == len(groups)

    def _get_block(**conv_params):
        layers = [nn.Conv2d(**conv_params)]
        if normalization is not None:
            normalization_fn = MODULES.get_if_str(normalization)
            layers.append(normalization_fn(conv_params["out_channels"]))
        if dropout_rate is not None:
            layers.append(nn.Dropout2d(p=dropout_rate))
        layers.append(activation_fn(inplace=True))
        return layers

    channels.insert(0, history_len * in_channels)
    params = []
    for i, (in_channels, out_channels) in enumerate(utils.pairwise(channels)):
        params.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "bias": use_bias,
                "kernel_size": kernel_sizes[i],
                "stride": strides[i],
                "groups": groups[i],
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


def get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    normalization: str = None,
    dropout_rate: float = None,
    activation: str = "ReLU",
    residual: Union[bool, str] = False,
    layer_order: List = None,
) -> nn.Module:

    features = features or [64, 128, 64]
    features.insert(0, history_len * in_features)

    net = SequentialNet(
        hiddens=features,
        layer_fn=nn.Linear,
        bias=use_bias,
        norm_fn=normalization,
        dropout=partial(nn.Dropout, p=dropout_rate)
        if dropout_rate is not None
        else None,
        activation_fn=activation,
        residual=residual,
        layer_order=layer_order,
    )

    return net
