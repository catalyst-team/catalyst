from typing import Dict  # isort:skip
from gym.spaces import Discrete
import torch
import torch.nn as nn

from catalyst.contrib.nn.modules import Flatten
from catalyst.rl.agent.head import ValueHead  # , StateNet
from catalyst.rl.core import CriticSpec, EnvironmentSpec
from catalyst.utils.initialization import create_optimal_inner_init


class ConvCritic(CriticSpec):
    """
    Critic that learns state value functions, like V(s).
    """

    def __init__(
        self,
        # state_net: StateNet,
        head_net: ValueHead,
    ):
        super().__init__()
        # self.state_net = state_net
        self.observation_net = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, groups=4),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, groups=4),
            # Flatten()
        )
        self.observation_net.apply(create_optimal_inner_init(nn.LeakyReLU))
        self.aggregation_net = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
        )
        self.aggregation_net.apply(create_optimal_inner_init(nn.LeakyReLU))
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
        x = state["image"]
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        x = x / 255.
        batch_size, *inner, h, w = x.shape
        x = x.view(batch_size, -1, h, w).squeeze_(2)
        # x = x.permute([0, 3, 1, 2])
        x = self.observation_net(x)
        # x = x.view(batch_size, history_len, -1)

        x = self.aggregation_net(x)

        x = self.head_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        # state_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # state_net = StateNet.get_from_params(**state_net_params)

        head_net = ValueHead(**value_head_params)

        net = cls(
            # state_net=state_net,
            head_net=head_net
        )

        return net


class ConvQCritic(ConvCritic):
    """
    Critic that learns state qvalue functions, like Q(s,a).
    """

    @classmethod
    def get_from_params(
        cls,
        # state_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        action_space = env_spec.action_space
        assert isinstance(action_space, Discrete)
        value_head_params["out_features"] = action_space.n
        net = super().get_from_params(value_head_params, env_spec)
        return net
