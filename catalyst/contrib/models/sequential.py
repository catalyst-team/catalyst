from typing import Union, List, Dict
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst.contrib.registry import MODULES
from catalyst.utils.misc import pairwise
from catalyst.utils.initialization import create_optimal_inner_init


def _process_additional_params(params, layers):
    if isinstance(params, List):
        assert len(params) == len(layers)
    else:
        params = [params] * len(layers)
    return params


class ResidualWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)


class SequentialNet(nn.Module):
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

        def _layer_fn(layer_fn, f_in, f_out, **kwargs):
            layer_fn = MODULES.get_if_str(layer_fn)
            layer_fn = layer_fn(f_in, f_out, **kwargs)
            return layer_fn

        def _normalization_fn(normalization_fn, f_in, f_out, **kwargs):
            normalization_fn = MODULES.get_if_str(normalization_fn)
            normalization_fn = \
                normalization_fn(f_out, **kwargs) \
                if normalization_fn is not None \
                else None
            return normalization_fn

        def _dropout_fn(dropout_fn, f_in, f_out, **kwargs):
            dropout_fn = MODULES.get_if_str(dropout_fn)
            dropout_fn = dropout_fn(**kwargs) \
                if dropout_fn is not None \
                else None
            return dropout_fn

        def _activation_fn(activation_fn, f_in, f_out, **kwargs):
            activation_fn = MODULES.get_if_str(activation_fn)
            activation_fn = activation_fn(**kwargs) \
                if activation_fn is not None \
                else None
            return activation_fn

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
        for i, (f_in, f_out) in enumerate(pairwise(hiddens)):
            block = []
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
                    block.append((f"{key}", sub_block))

            block_ = OrderedDict(block)
            block = torch.nn.Sequential(block_)

            if block_.get("act", None) is not None:
                activation = block_["act"]
                activation_init = \
                    create_optimal_inner_init(nonlinearity=activation)
                block.apply(activation_init)

            if residual == "hard" or (residual == "soft" and f_in == f_out):
                block = ResidualWrapper(net=block)
            net.append((f"block_{i}", block))

        self.net = torch.nn.Sequential(OrderedDict(net))

    def forward(self, x):
        x = self.net.forward(x)
        return x


__all__ = ["ResidualWrapper", "SequentialNet"]
