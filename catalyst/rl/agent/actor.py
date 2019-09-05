from typing import Dict
from gym.spaces import Box, Discrete

import torch

from catalyst.rl.core import ActorSpec, EnvironmentSpec
from .network import StateNet
from .head import PolicyHead


class Actor(ActorSpec):
    """
    Actor which learns agents policy.
    """

    def __init__(
        self,
        state_net: StateNet,
        head_net: PolicyHead,
    ):
        super().__init__()
        self.state_net = state_net
        self.head_net = head_net

    @property
    def policy_type(self) -> str:
        return self.head_net.policy_type

    def forward(self, state: torch.Tensor, logprob=False, deterministic=False):
        x = self.state_net(state)
        x = self.head_net(x, logprob, deterministic)
        return x

    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        policy_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: any better solution?
        action_space = env_spec.action_space
        if isinstance(action_space, Box):
            # continuous control
            policy_head_params["out_features"] = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            # discrete control
            policy_head_params["out_features"] = action_space.n
        else:
            raise NotImplementedError()

        # @TODO: any better solution?
        state_net_params["state_shape"] = env_spec.state_space.shape
        state_net = StateNet.get_from_params(**state_net_params)
        head_net = PolicyHead(**policy_head_params)

        net = cls(state_net=state_net, head_net=head_net)

        return net


__all__ = ["Actor"]
