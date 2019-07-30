from typing import Dict
from gym.spaces import Discrete

import torch

from catalyst.rl.agent.head import ValueHead
from catalyst.rl.core import CriticSpec, EnvironmentSpec

from .network import StateNet


class ConvCritic(CriticSpec):
    """
    Critic that learns state value functions, like V(s).
    """

    def __init__(self, state_net: StateNet, head_net: ValueHead):
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

    @property
    def num_heads(self) -> int:
        return self.head_net.num_heads

    @property
    def hyperbolic_constant(self) -> float:
        return self.head_net.hyperbolic_constant

    def forward(self, state: torch.Tensor):
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
        state_net = StateNet.get_from_params(**state_net_params)
        head_net = ValueHead(**value_head_params)

        net = cls(state_net=state_net, head_net=head_net)

        return net


class ConvQCritic(ConvCritic):
    """
    Critic that learns state qvalue functions, like Q(s,a).
    """

    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        action_space = env_spec.action_space
        assert isinstance(action_space, Discrete)
        value_head_params["out_features"] = action_space.n
        net = super().get_from_params(
            state_net_params, value_head_params, env_spec)
        return net
