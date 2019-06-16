from collections import OrderedDict

import torch
import torch.nn as nn

from catalyst.contrib.registry import MODULES
from catalyst.utils.misc import pairwise
from catalyst.utils.initialization import create_optimal_inner_init


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
        layer_fn=nn.Linear,
        bias=True,
        norm_fn=None,
        activation_fn=nn.ReLU,
        dropout=None,
        layer_order=None,
        residual=False
    ):
        super().__init__()
        assert len(hiddens) > 1, "No sequence found"

        layer_fn = MODULES.get_if_str(layer_fn)
        activation_fn = MODULES.get_if_str(activation_fn)
        norm_fn = MODULES.get_if_str(norm_fn)
        dropout = MODULES.get_if_str(dropout)
        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)

        layer_order = layer_order or ["layer", "norm", "drop", "act"]

        if isinstance(dropout, float):
            dropout_fn = lambda: nn.Dropout(dropout)
        else:
            dropout_fn = dropout

        def _layer_fn(f_in, f_out, bias):
            return layer_fn(f_in, f_out, bias=bias)

        def _normalize_fn(f_in, f_out, bias):
            return norm_fn(f_out) if norm_fn is not None else None

        def _dropout_fn(f_in, f_out, bias):
            return dropout_fn() if dropout_fn is not None else None

        def _activation_fn(f_in, f_out, bias):
            return activation_fn() if activation_fn is not None else None

        name2fn = {
            "layer": _layer_fn,
            "norm": _normalize_fn,
            "drop": _dropout_fn,
            "act": _activation_fn,
        }

        net = []

        for i, (f_in, f_out) in enumerate(pairwise(hiddens)):
            block = []
            for key in layer_order:
                fn = name2fn[key](f_in, f_out, bias)
                if fn is not None:
                    block.append((f"{key}", fn))
            block = torch.nn.Sequential(OrderedDict(block))
            if residual:
                block = ResidualWrapper(net=block)
            net.append((f"block_{i}", block))

        self.net = torch.nn.Sequential(OrderedDict(net))
        self.net.apply(inner_init)

    def forward(self, x):
        x = self.net.forward(x)
        return x


__all__ = ["ResidualWrapper", "SequentialNet"]
