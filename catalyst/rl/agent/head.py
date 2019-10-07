from typing import List

import torch
import torch.nn as nn

from catalyst.utils import outer_init
from catalyst.contrib.models import SequentialNet
from .policy import CategoricalPolicy, BernoulliPolicy, DiagonalGaussPolicy, \
    SquashingGaussPolicy, RealNVPPolicy


class ValueHead(nn.Module):
    @staticmethod
    def _build_head(in_features, out_features, num_atoms, bias):
        return nn.Linear(
            in_features=in_features,
            out_features=out_features * num_atoms,
            bias=bias
        )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_atoms: int = 1,
        use_state_value_head: bool = False,
        distribution: str = None,
        values_range: tuple = None,
        num_heads: int = 1,
        hyperbolic_constant: float = 1.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.num_atoms = num_atoms
        self.use_state_value_head = use_state_value_head

        self.distribution = distribution
        self.values_range = values_range

        self.num_heads = num_heads
        if self.num_heads == 1:
            hyperbolic_constant = 1.0
        self.hyperbolic_constant = hyperbolic_constant

        if distribution is None:  # mean case
            assert values_range is None and num_atoms == 1
        elif distribution == "categorical":
            assert values_range is not None and num_atoms > 1
        elif distribution == "quantile":
            assert values_range is None and num_atoms > 1
        else:
            raise NotImplementedError()

        value_heads = [
            self._build_head(in_features, out_features, num_atoms, bias)
            for _ in range(num_heads)
        ]
        self.value_heads = nn.ModuleList(value_heads)

        if self.use_state_value_head:
            assert self.out_features > 1, "Not implemented behaviour"
            state_value_heads = [
                self._build_head(in_features, 1, num_atoms, bias)
                for _ in range(num_heads)
            ]
            self.state_value_heads = nn.ModuleList(state_value_heads)

        self.apply(outer_init)

    def forward(self, state: torch.Tensor):
        x: List[torch.Tensor] = []
        for net in self.value_heads:
            x.append(net(state).view(-1, self.out_features, self.num_atoms))
        # batch_size(0) x num_heads(1) x num_outputs(2) x num_atoms(3)
        x = torch.stack(x, dim=1)

        if self.use_state_value_head:
            state_value: List[torch.Tensor] = []
            for net in self.state_value_heads:
                state_value.append(net(state).view(-1, 1, self.num_atoms))
            # batch_size(0) x num_heads(1) x num_outputs(2) x num_atoms(3)
            state_value = torch.stack(state_value, dim=1)

            x_mean = x.mean(2, keepdim=True)
            x = x - x_mean + state_value

        # batch_size(0) x num_heads(1) x num_outputs(2) x num_atoms(3)
        return x


class PolicyHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        policy_type: str = None,
        out_activation: nn.Module = None
    ):
        super().__init__()
        assert policy_type in [
            "categorical", "bernoulli", "diagonal-gauss",
            "squashing-gauss", "real-nvp",
            "logits", None
        ]

        # @TODO: refactor
        layer_fn = nn.Linear
        activation_fn = nn.ReLU
        squashing_fn = out_activation
        bias = True

        if policy_type == "categorical":
            assert out_activation is None
            head_size = out_features
            policy_net = CategoricalPolicy()
        elif policy_type == "bernoulli":
            assert out_activation is None
            head_size = out_features
            policy_net = BernoulliPolicy()
        elif policy_type == "diagonal-gauss":
            head_size = out_features * 2
            policy_net = DiagonalGaussPolicy()
        elif policy_type == "squashing-gauss":
            out_activation = None
            head_size = out_features * 2
            policy_net = SquashingGaussPolicy(squashing_fn)
        elif policy_type == "real-nvp":
            out_activation = None
            head_size = out_features * 2
            policy_net = RealNVPPolicy(
                action_size=out_features,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                squashing_fn=squashing_fn,
                bias=bias
            )
        else:
            head_size = out_features
            policy_net = None
            policy_type = "logits"

        self.policy_type = policy_type

        head_net = SequentialNet(
            hiddens=[in_features, head_size],
            layer_fn={"module": layer_fn, "bias": True},
            activation_fn=out_activation,
            norm_fn=None,
        )
        head_net.apply(outer_init)
        self.head_net = head_net

        self.policy_net = policy_net
        self._policy_fn = None
        if policy_net is not None:
            self._policy_fn = policy_net.forward
        else:
            self._policy_fn = lambda *args: args[0]

    def forward(self, state: torch.Tensor, logprob=None, deterministic=False):
        x = self.head_net(state)
        x = self._policy_fn(x, logprob, deterministic)
        return x


__all__ = ["ValueHead", "PolicyHead"]
