import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init
from catalyst.rl.agents.layers import StateNet, LamaPooling, PolicyHead
from catalyst.rl.registry import MODULES
from .core import ActorSpec


class Actor(ActorSpec):
    """
    Actor which learns agents policy.
    """

    def __init__(
        self,
        main_net: nn.Module,
        head_net: PolicyHead = None,
        observation_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        super().__init__()
        self.representation_net = StateNet(
            main_net=main_net,
            observation_net=observation_net,
            aggregation_net=aggregation_net
        )
        self.head_net = head_net

    def forward(self, state, with_log_pi=False, deterministic=False):
        x = self.representation_net(state)
        x = self.head_net(x, with_log_pi, deterministic)
        return x

    @property
    def policy_type(self) -> str:
        return self.head_net.policy_type

    @classmethod
    def create_from_params(
        cls,
        state_shape,
        action_size,
        observation_hiddens=None,
        head_hiddens=None,
        layer_fn=nn.Linear,
        activation_fn=nn.ReLU,
        dropout=None,
        norm_fn=None,
        bias=True,
        layer_order=None,
        residual=False,
        out_activation=None,
        observation_aggregation=None,
        lama_poolings=None,
        policy_type=None,
    ):

        observation_hiddens = observation_hiddens or []
        head_hiddens = head_hiddens or []

        layer_fn = MODULES.get_if_str(layer_fn)
        activation_fn = MODULES.get_if_str(activation_fn)
        norm_fn = MODULES.get_if_str(norm_fn)
        out_activation = MODULES.get_if_str(out_activation)
        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            # state_shape like [history_len, obs_shape]
            # @TODO: handle lama/rnn correctly
            if not observation_aggregation:
                observation_size = reduce(lambda x, y: x * y, state_shape)
            else:
                observation_size = reduce(lambda x, y: x * y, state_shape[1:])

            if len(observation_hiddens) > 0:
                observation_net = SequentialNet(
                    hiddens=[observation_size] + observation_hiddens,
                    layer_fn=layer_fn,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    norm_fn=norm_fn,
                    bias=bias,
                    layer_order=layer_order,
                    residual=residual
                )
                observation_net.apply(inner_init)
                obs_out = observation_hiddens[-1]
            else:
                observation_net = None
                obs_out = observation_size

        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        assert obs_out

        if observation_aggregation == "lama_obs":
            aggregation_net = LamaPooling(
                features_in=obs_out,
                poolings=lama_poolings
            )
            aggregation_out = aggregation_net.features_out
        else:
            aggregation_net = None
            aggregation_out = obs_out

        main_net = SequentialNet(
            hiddens=[aggregation_out] + head_hiddens,
            layer_fn=layer_fn,
            dropout=dropout,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            layer_order=layer_order,
            residual=residual
        )
        main_net.apply(inner_init)

        # @TODO: place for memory network

        head_net = PolicyHead(
            in_features=head_hiddens[-1],
            out_features=action_size,
            policy_type=policy_type,
            out_activation=out_activation
        )

        actor_net = cls(
            observation_net=observation_net,
            aggregation_net=aggregation_net,
            main_net=main_net,
            head_net=head_net,
        )

        return actor_net
