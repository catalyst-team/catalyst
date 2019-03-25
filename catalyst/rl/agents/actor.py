import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.layers import StateNet, \
    GaussPolicy, RealNVPPolicy, \
    LamaPooling
from catalyst.rl.registry import MODULES


class Actor(StateNet):
    """
    Actor which learns deterministic policy.
    """

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
        squashing_fn=nn.Tanh,
        **kwargs
    ):
        assert len(kwargs) == 0

        observation_hiddens = observation_hiddens or []
        head_hiddens = head_hiddens or []

        layer_fn = MODULES.get(layer_fn)
        activation_fn = MODULES.get(activation_fn)
        norm_fn = MODULES.get(norm_fn)
        out_activation = MODULES.get(out_activation)
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

        if policy_type == "gauss":
            head_size = action_size * 2
            policy_net = GaussPolicy(squashing_fn)
        elif policy_type == "real_nvp":
            head_size = action_size * 2
            policy_net = RealNVPPolicy(
                action_size=action_size,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                squashing_fn=squashing_fn,
                norm_fn=None,
                bias=bias
            )
        else:
            head_size = action_size
            policy_net = None

        head_net = SequentialNet(
            hiddens=[head_hiddens[-1], head_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )
        head_net.apply(outer_init)

        actor_net = cls(
            observation_net=observation_net,
            aggregation_net=aggregation_net,
            main_net=main_net,
            head_net=head_net,
            policy_net=policy_net
        )

        return actor_net
