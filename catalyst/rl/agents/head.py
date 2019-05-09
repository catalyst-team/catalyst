import torch
import torch.nn as nn
from catalyst.dl.initialization import outer_init
from catalyst.contrib.models import SequentialNet
from .policy import CategoricalPolicy, GaussPolicy, RealNVPPolicy


class ValueHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_atoms: int = 1,
        bias: bool = False,
        distribution: str = None,
        values_range: tuple = None
    ):
        super().__init__()

        self.out_features = out_features
        self.num_atoms = num_atoms
        self.distribution = distribution
        self.values_range = values_range

        if distribution is None:  # mean case
            assert values_range is None and num_atoms == 1
        elif distribution == "categorical":
            assert values_range is not None and num_atoms > 1
        elif distribution == "quantile":
            assert values_range is None and num_atoms > 1
        else:
            raise NotImplementedError()

        self.net = nn.Linear(
            in_features=in_features,
            out_features=out_features * num_atoms,
            bias=bias
        )
        self.apply(outer_init)

    def forward(self, inputs):
        x: torch.Tensor = \
            self.net(inputs).view(-1, self.out_features, self.num_atoms)
        # x = x.squeeze_(dim=-1)
        x = x.squeeze_(dim=1).squeeze_(dim=-1)
        if self.num_atoms == 1 and self.out_features == 1:
            # make critic outputs (B, 1) instead of (B, )
            x = x.unsqueeze_(dim=1)
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
            "categorical", "gauss", "real_nvp",
            "logits", None
        ]

        # @TODO: refactor
        layer_fn = nn.Linear
        activation_fn = nn.ReLU
        squashing_fn = nn.Tanh
        bias = True

        if policy_type == "categorical":
            head_size = out_features
            policy_net = CategoricalPolicy()
        elif policy_type == "gauss":
            head_size = out_features * 2
            policy_net = GaussPolicy(squashing_fn)
        elif policy_type == "real_nvp":
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
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )
        head_net.apply(outer_init)
        self.head_net = head_net

        self.policy_net = policy_net
        self._policy_fn = None
        if policy_net is None:
            self._policy_fn = lambda *args: args[0]
        elif isinstance(
                policy_net,
                (CategoricalPolicy, GaussPolicy, RealNVPPolicy)):
            self._policy_fn = policy_net.forward
        else:
            raise NotImplementedError

    def forward(self, inputs, logprob=False, deterministic=False):
        x = self.head_net(inputs)
        x = self._policy_fn(x, logprob, deterministic)
        return x
