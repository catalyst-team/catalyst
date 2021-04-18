# Author: Sergey Kolesnikov, scitator@gmail.com
# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, Union
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from catalyst import utils
from catalyst.registry import REGISTRY


def _process_additional_params(params, layers):
    if isinstance(params, List):
        assert len(params) == len(layers)
    else:
        params = [params] * len(layers)
    return params


def _layer_fn(layer_fn, f_in, f_out, **kwargs):
    layer_fn = REGISTRY.get_if_str(layer_fn)
    layer_fn = layer_fn(f_in, f_out, **kwargs)
    return layer_fn


def _normalization_fn(normalization_fn, f_in, f_out, **kwargs):
    normalization_fn = REGISTRY.get_if_str(normalization_fn)
    normalization_fn = normalization_fn(f_out, **kwargs) if normalization_fn is not None else None
    return normalization_fn


def _dropout_fn(dropout_fn, f_in, f_out, **kwargs):
    dropout_fn = REGISTRY.get_if_str(dropout_fn)
    dropout_fn = dropout_fn(**kwargs) if dropout_fn is not None else None
    return dropout_fn


def _activation_fn(activation_fn, f_in, f_out, **kwargs):
    activation_fn = REGISTRY.get_if_str(activation_fn)
    activation_fn = activation_fn(**kwargs) if activation_fn is not None else None
    return activation_fn


class ResidualWrapper(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, net):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.net = net

    def forward(self, x):
        """Forward call."""
        return x + self.net(x)


class SequentialNet(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        hiddens,
        layer_fn: Union[str, Dict, List],
        norm_fn: Union[str, Dict, List] = None,
        dropout_fn: Union[str, Dict, List] = None,
        activation_fn: Union[str, Dict, List] = None,
        residual: Union[bool, str] = False,
        layer_order: List = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        assert len(hiddens) > 1, "No sequence found"

        # layer params
        layer_fn = _process_additional_params(layer_fn, hiddens[1:])
        # normalization params
        norm_fn = _process_additional_params(norm_fn, hiddens[1:])
        # dropout params
        dropout_fn = _process_additional_params(dropout_fn, hiddens[1:])
        # activation params
        activation_fn = _process_additional_params(activation_fn, hiddens[1:])

        if isinstance(residual, bool) and residual:
            residual = "hard"
            residual = _process_additional_params(residual, hiddens[1:])

        layer_order = layer_order or ["layer", "norm", "drop", "act"]

        name2fn = {
            "layer": _layer_fn,
            "norm": _normalization_fn,
            "drop": _dropout_fn,
            "act": _activation_fn,
        }
        name2params = {
            "layer": layer_fn,
            "norm": norm_fn,
            "drop": dropout_fn,
            "act": activation_fn,
        }

        net = []
        for i, (f_in, f_out) in enumerate(utils.pairwise(hiddens)):
            block_list = []
            for key in layer_order:
                sub_fn = name2fn[key]
                sub_params = deepcopy(name2params[key][i])

                if isinstance(sub_params, Dict):
                    sub_module = sub_params.pop("module")
                else:
                    sub_module = sub_params
                    sub_params = {}

                sub_block = sub_fn(sub_module, f_in, f_out, **sub_params)
                if sub_block is not None:
                    block_list.append((f"{key}", sub_block))

            block_dict = OrderedDict(block_list)
            block_net = torch.nn.Sequential(block_dict)

            if block_dict.get("act", None) is not None:
                activation = block_dict["act"]
                activation_init = utils.get_optimal_inner_init(nonlinearity=activation)
                block_net.apply(activation_init)

            if residual == "hard" or (residual == "soft" and f_in == f_out):
                block_net = ResidualWrapper(net=block_net)
            net.append((f"block_{i}", block_net))

        self.net = torch.nn.Sequential(OrderedDict(net))

    def forward(self, x):
        """Forward call."""
        x = self.net.forward(x)
        return x


__all__ = ["ResidualWrapper", "SequentialNet"]
