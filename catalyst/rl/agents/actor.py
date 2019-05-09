from typing import Dict
from functools import reduce
from gym.spaces import Box, Discrete

from catalyst.rl.environments.core import EnvironmentSpec
from .core import ActorSpec
from .net import StateNet
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

    def forward(self, state, logprob=False, deterministic=False):
        x = self.state_net(state)
        x = self.head_net(x, logprob, deterministic)
        return x

    @property
    def policy_type(self) -> str:
        return self.head_net.policy_type

    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        policy_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        # @TODO: refactor
        observation_size = reduce(
            lambda x, y: x * y,
            env_spec.state_space.shape)

        state_net_params["observation_net_params"]["hiddens"].insert(
            0, observation_size)

        # @TODO: any better solution?
        action_space = env_spec.action_space
        if isinstance(action_space, Box):
            policy_head_params["out_features"] = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            policy_head_params["out_features"] = action_space.n
        else:
            raise NotImplementedError()

        # @TODO: make by init?
        state_net = StateNet.get_from_params(**state_net_params)
        head_net = PolicyHead(**policy_head_params)

        net = cls(
            state_net=state_net,
            head_net=head_net
        )

        return net
