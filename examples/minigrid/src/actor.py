from typing import Dict  # isort:skip
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

from catalyst.contrib.nn.modules import Flatten
from catalyst.rl.agent.head import PolicyHead  # , StateNet
from catalyst.rl.core import ActorSpec, EnvironmentSpec
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
    def policy_type(self) -> str:
        return self.head_net.policy_type

    def forward(self, state: torch.Tensor, logprob=False, deterministic=False):
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
            policy_head_params["features_out"] = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            policy_head_params["features_out"] = action_space.n
        else:
            raise NotImplementedError()

        # state_net = StateNet.get_from_params(**state_net_params)
        head_net = PolicyHead(**policy_head_params)

        net = cls(
            # state_net=state_net,
            head_net=head_net
        )

        return net
