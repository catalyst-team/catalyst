from typing import Dict
import torch.nn as nn
from functools import reduce
from gym.spaces import Discrete

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.layers import StateNet, StateActionNet, \
    LamaPooling, ValueHead
from ..registry import MODULES
from .core import CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


class StateCritic(CriticSpec):
    """
    Critic that learns state value functions, like V(s).
    """

    def __init__(
        self,
        state_net: StateNet,
        head_net: ValueHead,
    ):
        super().__init__()
        self.state_net = state_net
        self.head_net = head_net

    def forward(self, state):
        x = self.state_net(state)
        x = self.head_net(x)
        return x

    @property
    def num_outputs(self) -> int:
        return self.head_net.out_features

    @property
    def num_atoms(self) -> int:
        return self.head_net.num_atoms

    @property
    def distribution(self) -> str:
        return self.head_net.distribution

    @property
    def values_range(self) -> tuple:
        return self.head_net.values_range

    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: refactor
        observation_size = reduce(
            lambda x, y: x * y,
            env_spec.state_space.shape)

        state_net_params["encoder_net_params"]["hiddens"].insert(
            0, observation_size)

        # @TODO: make by init?
        state_net = StateNet.get_from_params(**state_net_params)
        head_net = ValueHead(**value_head_params)

        critic_net = cls(
            state_net=state_net,
            head_net=head_net
        )

        return critic_net


class ActionCritic(StateCritic):
    """
    Critic that learns state-action value functions, like Q(s).
    """

    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: any better solution?
        action_space = env_spec.action_space
        assert isinstance(action_space, Discrete)
        value_head_params["out_features"] = action_space.n
        critic = super().get_from_params(
            state_net_params=state_net_params,
            value_head_params=value_head_params,
            env_spec=env_spec
        )
        return critic


class StateActionCritic(CriticSpec):
    """
    Critic which learns state-action value functions, like Q(s, a).
    """

    def __init__(
        self,
        main_net: nn.Module,
        head_net: ValueHead,
        observation_net: nn.Module = None,
        action_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        super().__init__()
        self.representation_net = StateActionNet(
            main_net=main_net,
            observation_net=observation_net,
            action_net=action_net,
            aggregation_net=aggregation_net
        )
        self.head_net = head_net

    def forward(self, state, action):
        x = self.representation_net(state, action)
        x = self.head_net(x)
        return x

    @property
    def num_outputs(self) -> int:
        return self.head_net.out_features

    @property
    def num_atoms(self) -> int:
        return self.head_net.num_atoms

    @classmethod
    def get_from_params(
        cls,
        state_shape,
        action_size,
        num_atoms=1,
        observation_hiddens=None,
        action_hiddens=None,
        head_hiddens=None,
        layer_fn=nn.Linear,
        activation_fn=nn.ReLU,
        dropout=None,
        norm_fn=None,
        bias=True,
        layer_order=None,
        residual=False,
        observation_aggregation=None,
        lama_poolings=None,
    ):

        observation_hiddens = observation_hiddens or []
        action_hiddens = action_hiddens or []
        head_hiddens = head_hiddens or []

        layer_fn = MODULES.get_if_str(layer_fn)
        activation_fn = MODULES.get_if_str(activation_fn)
        norm_fn = MODULES.get_if_str(norm_fn)
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
            act_out = action_hiddens[-1]
        else:
            action_net = None
            act_out = action_size

        assert obs_out and act_out

        if observation_aggregation == "lama_obs":
            aggregation_net = LamaPooling(
                features_in=obs_out,
                poolings=lama_poolings
            )
            aggregation_out = aggregation_net.features_out + act_out
        else:
            aggregation_net = None
            aggregation_out = obs_out + act_out

        main_net = SequentialNet(
            hiddens=[aggregation_out] + head_hiddens[:-2],
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

        head_net = ValueHead(
            in_features=head_hiddens[-2],
            out_features=head_hiddens[-1],
            num_atoms=num_atoms
        )
        head_net.apply(outer_init)

        critic_net = cls(
            observation_net=observation_net,
            action_net=action_net,
            aggregation_net=aggregation_net,
            main_net=main_net,
            head_net=head_net
        )

        return critic_net
