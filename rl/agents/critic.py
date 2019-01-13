import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.layers import StateNet, StateActionNet


class Critic(StateActionNet):
    """
    Critic which learns state-action value function Q(s,a).
    """

    @classmethod
    def create_from_config(
        cls,
        state_shape,
        action_size,
        observation_hiddens,
        action_hiddens,
        head_hiddens,
        layer_fn,
        activation_fn=nn.ReLU,
        dropout=None,
        norm_fn=None,
        bias=True,
        layer_order=None,
        residual=False,
        out_activation=None,
        memory_type=None,
        **kwargs
    ):
        assert len(kwargs) == 0
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            state_size = reduce(lambda x, y: x * y, state_shape)

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
        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        action_net = SequentialNet(
            hiddens=[action_size] + action_hiddens,
            layer_fn=layer_fn,
            dropout=dropout,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            layer_order=layer_order,
            residual=residual
        )

        if memory_type == "lama":
            raise NotImplementedError
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None
            memory_out = observation_hiddens[-1] + action_hiddens[-1]

        head_net = SequentialNet(
            hiddens=[memory_out] + head_hiddens,
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        observation_net.apply(inner_init)
        action_net.apply(inner_init)
        head_net.apply(outer_init)

        critic_net = cls(
            observation_net=observation_net,
            action_net=action_net,
            memory_net=memory_net,
            head_net=head_net
        )

        return critic_net


class ValueCritic(StateNet):
    """
    Critic which learns state-action value function Q(s,a).
    """

    @classmethod
    def create_from_config(
        cls,
        state_shape,
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
        **kwargs
    ):
        assert len(kwargs) == 0
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            state_size = reduce(lambda x, y: x * y, state_shape)

            observation_net = SequentialNet(
                hiddens=[state_size] + hiddens[:-1],
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
            raise NotImplementedError
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None
            memory_out = hiddens[-2]

        head_net = SequentialNet(
            hiddens=[memory_out, hiddens[-1]],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        observation_net.apply(inner_init)
        head_net.apply(outer_init)

        critic_net = cls(
            observation_net=observation_net,
            memory_net=memory_net,
            head_net=head_net
        )

        return critic_net
