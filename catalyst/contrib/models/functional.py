# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List, Union

from torch import nn

from catalyst.contrib.models.sequential import (  # noqa: WPS450
    _process_additional_params,
    SequentialNet,
)


def get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    kernel_sizes: List = None,
    strides: List = None,
    groups: List = None,
    use_bias: Union[bool, List] = False,
    normalization: Union[str, List] = None,
    dropout_rate: Union[float, List] = None,
    activation: Union[str, List] = None,
    residual: Union[bool, str] = False,
    layer_order: List = None,
) -> nn.Module:
    """@TODO: Docs. Contribution is welcome."""
    channels = channels or [32, 64, 64]
    kernel_sizes = kernel_sizes or [8, 4, 3]
    strides = strides or [4, 2, 1]
    groups = groups or [1, 1, 1]
    assert len(channels) == len(kernel_sizes) == len(strides) == len(groups)
    use_bias = _process_additional_params(use_bias, channels)

    layer_fn = [
        {
            "module": nn.Conv2d,
            "bias": bias,
            "kernel_size": kernel_size,
            "stride": stride,
            "groups": group,
        }
        for bias, kernel_size, stride, group in zip(use_bias, kernel_sizes, strides, groups)
    ]

    if dropout_rate is not None:
        dropout_fn = (
            {"module": nn.Dropout2d, "p": dropout_rate}
            if isinstance(dropout_rate, float)
            else [
                {"module": nn.Dropout2d, "p": p} if p is not None else None for p in dropout_rate
            ]
        )
    else:
        dropout_fn = None

    channels.insert(0, history_len * in_channels)
    net = SequentialNet(
        hiddens=channels,
        layer_fn=layer_fn,
        norm_fn=normalization,
        dropout_fn=dropout_fn,
        activation_fn=activation,
        residual=residual,
        layer_order=layer_order,
    )
    return net


def get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: Union[bool, List] = False,
    normalization: Union[str, List] = None,
    dropout_rate: Union[float, List] = None,
    activation: Union[str, List] = None,
    residual: Union[bool, str] = False,
    layer_order: List = None,
) -> nn.Module:
    """@TODO: Docs. Contribution is welcome."""
    features = features or [64, 128, 64]

    layer_fn = (
        {"module": nn.Linear, "bias": use_bias}
        if isinstance(use_bias, bool)
        else [{"module": nn.Linear, "bias": bias} for bias in use_bias]
    )
    if dropout_rate is not None:
        dropout_fn = (
            {"module": nn.Dropout, "p": dropout_rate}
            if isinstance(dropout_rate, float)
            else [{"module": nn.Dropout, "p": p} if p is not None else None for p in dropout_rate]
        )
    else:
        dropout_fn = None

    features.insert(0, history_len * in_features)
    net = SequentialNet(
        hiddens=features,
        layer_fn=layer_fn,
        norm_fn=normalization,
        dropout_fn=dropout_fn,
        activation_fn=activation,
        residual=residual,
        layer_order=layer_order,
    )

    return net


__all__ = ["get_convolution_net", "get_linear_net"]
