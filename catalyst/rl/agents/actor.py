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
        hiddens,
        layer_fn,
        activation_fn=nn.ReLU,
        dropout=None,
        norm_fn=None,
        bias=True,
        layer_order=None,
        residual=False,
        out_activation=None,
        memory_type=None,
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

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            # state_shape like [history_len, obs_shape]
            # @TODO: handle lama correctly
            if not memory_type:
                state_size = reduce(lambda x, y: x * y, state_shape)
            else:
                state_size = reduce(lambda x, y: x * y, state_shape[1:])

            observation_net = SequentialNet(
                hiddens=[state_size] + hiddens,
                layer_fn=layer_fn,
                dropout=dropout,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias,
                layer_order=layer_order,
                residual=residual
            )
        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        if memory_type == "lama":
            memory_net = LamaPooling(
                features_in=hiddens[-1],
                poolings=lama_poolings
            )
            memory_out = memory_net.features_out
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None
            memory_out = hiddens[-1]

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
            hiddens=[memory_out, head_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        observation_net.apply(inner_init)
        head_net.apply(outer_init)

        actor_net = cls(
            observation_net=observation_net,
            memory_net=memory_net,
            head_net=head_net,
            policy_net=policy_net
        )

        return actor_net
