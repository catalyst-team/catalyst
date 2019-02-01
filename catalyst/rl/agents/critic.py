import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.layers import StateNet, StateActionNet, LamaPooling


class Critic(StateActionNet):
    """
    Critic which learns state-action value function Q(s,a).
    """

    @classmethod
    def create_from_params(
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
        lama_poolings=None,
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
            # @TODO: handle lama correctly
            if not memory_type:
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
            else:
                observation_net = None

        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        if len(action_hiddens) > 0:
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
            action_net.apply(inner_init)
        else:
            action_net = None

        if memory_type == "lama":
            memory_net = LamaPooling(
                features_in=observation_hiddens[-1],
                poolings=lama_poolings
            )
            memory_out = memory_net.features_out + action_hiddens[-1]
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None

            if len(observation_hiddens) + len(action_hiddens) == 0:
                memory_out = state_size + action_size
            else:
                # @TODO: do a normal fix
                memory_out = observation_hiddens[-1] + action_hiddens[-1]

        bone_net = SequentialNet(
            hiddens=[memory_out] + head_hiddens[:-1],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            layer_order=layer_order,
            residual=residual
        )
        bone_net.apply(inner_init)

        head_net = SequentialNet(
            hiddens=[head_hiddens[-2], head_hiddens[-1]],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )
        head_net.apply(outer_init)

        critic_net = cls(
            observation_net=observation_net,
            action_net=action_net,
            memory_net=memory_net,
            bone_net=bone_net,
            head_net=head_net
        )

        return critic_net


class ValueCritic(StateNet):
    """
    Critic which learns value function V(s).
    """

    @classmethod
    def create_from_params(
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
        lama_poolings=None,
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
            state_size = reduce(lambda x, y: x * y, state_shape)

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
            memory_out = memory_net.features_out + hiddens[-1]
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None
            memory_out = hiddens[-1]

        head_net = SequentialNet(
            hiddens=[memory_out, 1],
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
            head_net=head_net,
            policy_net=None
        )

        return critic_net
