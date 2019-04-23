from typing import Dict
from functools import reduce
from gym.spaces import Discrete

from catalyst.rl.agents.layers import StateNet, StateActionNet, ValueHead
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

    def forward(self, state):
        x = self.state_net(state)
        x = self.head_net(x)
        return x

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

        state_net_params["observation_net_params"]["hiddens"].insert(
            0, observation_size)

        # @TODO: make by init?
        state_net = StateNet.get_from_params(**state_net_params)
        head_net = ValueHead(**value_head_params)

        net = cls(
            state_net=state_net,
            head_net=head_net
        )

        return net


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
        net = super().get_from_params(
            state_net_params=state_net_params,
            value_head_params=value_head_params,
            env_spec=env_spec
        )
        return net


class StateActionCritic(CriticSpec):
    """
    Critic which learns state-action value functions, like Q(s, a).
    """

    def __init__(
        self,
        state_action_net: StateActionNet,
        head_net: ValueHead,
    ):
        super().__init__()
        self.state_action_net = state_action_net
        self.head_net = head_net

    def forward(self, state, action):
        x = self.state_action_net(state, action)
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
        state_action_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: refactor
        observation_size = reduce(
            lambda x, y: x * y,
            env_spec.state_space.shape)
        state_action_net_params["observation_net_params"]["hiddens"]\
            .insert(0, observation_size)

        action_size = reduce(
            lambda x, y: x * y,
            env_spec.action_space.shape)
        state_action_net_params["action_net_params"]["hiddens"] \
            .insert(0, action_size)

        value_head_params["out_features"] = 1

        # @TODO: make by init?
        state_action_net = StateActionNet.get_from_params(
            **state_action_net_params)
        head_net = ValueHead(**value_head_params)

        net = cls(
            state_action_net=state_action_net,
            head_net=head_net
        )

        return net
