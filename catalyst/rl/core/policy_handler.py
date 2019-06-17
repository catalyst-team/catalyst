from typing import Union
import numpy as np
from gym.spaces import Box, Discrete
import torch

from .agent import ActorSpec, CriticSpec
from .environment import EnvironmentSpec


class PolicyHandler:
    def __init__(
        self, env: EnvironmentSpec, agent: Union[ActorSpec, CriticSpec], device
    ):
        self.action_fn = None
        self.discrete_actions = isinstance(env.action_space, Discrete)

        # DQN, PPO
        if self.discrete_actions:
            if isinstance(agent, ActorSpec):
                self.action_clip = None
                self.action_fn = self._actor_handler
            elif isinstance(agent, CriticSpec):
                self.action_fn = self._critic_handler
                self.value_distribution = agent.distribution
                if self.value_distribution == "categorical":
                    v_min, v_max = agent.values_range
                    self.z = torch.linspace(
                        start=v_min, end=v_max, steps=agent.num_atoms
                    ).to(device)
        # DDPG, SAC, TD3
        else:
            assert isinstance(agent, ActorSpec)
            action_space: Box = env.action_space
            self.action_clip = action_space.low, action_space.high
            self.action_fn = self._actor_handler

    @torch.no_grad()
    def _get_q_values(self, critic: CriticSpec, state: np.ndarray, device):
        states = torch.Tensor(state).to(device).unsqueeze(0)
        output = critic(states)
        # We use the last head to perform actions
        # This is the head corresponding to the largest gamma
        if self.value_distribution == "categorical":
            probs = torch.softmax(output[0, -1, :, :], dim=-1)
            q_values = torch.sum(probs * self.z, dim=-1)
        elif self.value_distribution == "quantile":
            q_values = torch.mean(output[0, -1, :, :], dim=-1)
        else:
            q_values = output[0, -1, :]
        return q_values.cpu().numpy()

    @torch.no_grad()
    def _sample_from_actor(
        self,
        actor: ActorSpec,
        state: np.ndarray,
        device,
        deterministic: bool = False
    ):
        states = torch.Tensor(state).to(device).unsqueeze(0)
        action = actor(states, deterministic=deterministic)
        action = action[0].cpu().numpy()

        if self.action_clip is not None:
            action = np.clip(
                action, a_min=self.action_clip[0], a_max=self.action_clip[1]
            )
        return action

    def _critic_handler(
        self,
        agent: CriticSpec,
        state: np.ndarray,
        device,
        deterministic: bool = False,
        exploration_strategy=None
    ):
        q_values = self._get_q_values(agent, state, device)
        action = exploration_strategy.get_action(q_values)
        return action

    def _actor_handler(
        self,
        agent: ActorSpec,
        state: np.ndarray,
        device,
        deterministic: bool = False,
        exploration_strategy=None
    ):
        action = self._sample_from_actor(agent, state, device, deterministic)
        action = exploration_strategy.get_action(action)
        return action
