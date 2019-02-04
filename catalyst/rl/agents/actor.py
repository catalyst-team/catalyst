import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.layers import StateNet, \
    GaussPolicy, RealNVPPolicy, \
    LamaPooling


class Actor(StateNet):
    """
    Actor which learns deterministic policy.
    """

    @classmethod
    def create_from_params(
        cls,
        state_shape,
        action_size,
        observation_hiddens,
        head_hiddens,
        layer_fn,
        activation_fn=nn.ReLU,
        dropout=None,
        norm_fn=None,
        bias=True,
        layer_order=None,
        residual=False,
        out_activation=None,
        history_aggregation_type=None,
        lama_poolings=None,
        policy_type=None,
        squashing_fn=nn.Tanh,
        **kwargs
    ):
        assert len(kwargs) == 0
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry

        layer_fn = Registry.name2nn(layer_fn)
        activation_fn = Registry.name2nn(activation_fn)
        norm_fn = Registry.name2nn(norm_fn)
        out_activation = Registry.name2nn(out_activation)
        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            # state_shape like [history_len, obs_shape]
            # @TODO: handle lama/rnn correctly
            if not history_aggregation_type:
                state_size = reduce(lambda x, y: x * y, state_shape)
            else:
                state_size = reduce(lambda x, y: x * y, state_shape[1:])

            if len(observation_hiddens) > 0:
                observation_net = SequentialNet(
                    hiddens=[state_size] + observation_hiddens,
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
                obs_out = state_size

        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        assert obs_out

        if history_aggregation_type == "lama_obs":
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
