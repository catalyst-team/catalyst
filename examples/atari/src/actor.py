from typing import Dict
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from catalyst.contrib.modules import Flatten

from catalyst.rl.agents.head import PolicyHead  # , StateNet
from catalyst.rl.agents import ActorSpec
from catalyst.rl.environments import EnvironmentSpec
from catalyst.utils.initialization import create_optimal_inner_init


class ConvActor(ActorSpec):
    def __init__(
        self,
        # state_net: StateNet,
        head_net: PolicyHead,
    ):
        super().__init__()
        # self.state_net = state_net
        self.observation_net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=4),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=4, groups=4),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=4),
            # Flatten()
        )
        self.observation_net.apply(create_optimal_inner_init(nn.LeakyReLU))
        self.aggregation_net = nn.Sequential(
            Flatten(),
            nn.Linear(576, 512),
            nn.LayerNorm(512),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
        )
        self.aggregation_net.apply(create_optimal_inner_init(nn.LeakyReLU))
        self.head_net = head_net

    @property
    def policy_type(self) -> str:
        return self.head_net.policy_type

    def forward(self, state: torch.Tensor, logprob=False, deterministic=False):
        x = state
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        x = x / 255.
        batch_size, history_len, *feature_size = x.shape
        x = x.view(-1, history_len, *feature_size).squeeze_(2)
        # x = x.permute([0, 3, 1, 2])
        x = self.observation_net(x)
        # x = x.view(batch_size, history_len, -1)

        x = self.aggregation_net(x)

        x = self.head_net(x, logprob, deterministic)
        return x

    @classmethod
    def get_from_params(
        cls,
        # state_net_params: Dict,
        policy_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: any better solution?
        action_space = env_spec.action_space
        if isinstance(action_space, Box):
            policy_head_params["out_features"] = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            policy_head_params["out_features"] = action_space.n
        else:
            raise NotImplementedError()

        # state_net = StateNet.get_from_params(**state_net_params)
        head_net = PolicyHead(**policy_head_params)

        net = cls(
            # state_net=state_net,
            head_net=head_net
        )

        return net
