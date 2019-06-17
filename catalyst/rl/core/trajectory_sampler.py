from typing import Union
import numpy as np
from ctypes import c_bool
import multiprocessing as mp
from dynarray import DynamicArray
import torch

from .agent import ActorSpec, CriticSpec
from .environment import EnvironmentSpec
from .policy_handler import PolicyHandler


class TrajectorySampler:
    def __init__(
        self,
        env: EnvironmentSpec,
        agent: Union[ActorSpec, CriticSpec],
        device,
        deterministic: bool = False,
        initial_capacity: int = int(1e3),
        sample_flag: mp.Value = None
    ):
        self.env = env
        self.agent = agent
        self._device = device
        self.deterministic = deterministic
        self.initial_capacity = initial_capacity
        self.policy_handler = PolicyHandler(
            env=self.env, agent=self.agent, device=device
        )

        self._sample_flag = sample_flag or mp.Value(c_bool, True)
        self._init_buffers()

    def _init_buffers(self):
        self.observations = DynamicArray(
            array_or_shape=(None, ) + tuple(self.env.observation_space.shape),
            dtype=self.env.observation_space.dtype,
            capacity=int(self.initial_capacity)
        )
        self.actions = DynamicArray(
            array_or_shape=(None, ) + tuple(self.env.action_space.shape),
            dtype=self.env.action_space.dtype,
            capacity=int(self.initial_capacity)
        )
        self.rewards = DynamicArray(
            array_or_shape=(None, ),
            dtype=np.float32,
            capacity=int(self.initial_capacity)
        )
        self.dones = DynamicArray(
            array_or_shape=(None, ),
            dtype=np.bool,
            capacity=int(self.initial_capacity)
        )

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def _init_with_observation(self, observation):
        self.observations.append(observation)

    def _put_transition(self, transition):
        """
        transition = [o_tp1, a_t, r_t, d_t]
        """
        o_tp1, a_t, r_t, d_t = transition
        self.observations.append(o_tp1)
        self.actions.append(a_t)
        self.rewards.append(r_t)
        self.dones.append(d_t)

    def _get_states_history(self, history_len=None):
        history_len = history_len or self.env.history_len
        states = [
            self.get_state(history_len=history_len, index=i)
            for i in range(len(self.observations))
        ]
        states = np.array(states)
        return states

    def get_state(self, index=None, history_len=None):
        index = index if index is not None else len(self.observations) - 1
        history_len = history_len \
            if history_len is not None \
            else self.env.history_len

        state = np.zeros(
            (history_len, ) + self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )

        indices = np.arange(max(0, index - history_len + 1), index + 1)
        state[-len(indices):] = self.observations[indices]
        return state

    def get_trajectory(self):
        trajectory = (
            np.array(self.observations[:-1]), np.array(self.actions),
            np.array(self.rewards), np.array(self.dones)
        )
        return trajectory

    @torch.no_grad()
    def reset(self, exploration_strategy=None):

        from catalyst.rl.exploration import \
            ParameterSpaceNoise, OrnsteinUhlenbeckProcess

        if isinstance(exploration_strategy, OrnsteinUhlenbeckProcess):
            exploration_strategy.reset_state(self.env.action_space.shape[0])

        if isinstance(exploration_strategy, ParameterSpaceNoise) \
                and len(self.observations) > 1:
            states = self._get_states_history()
            states = self._to_tensor(states)
            exploration_strategy.update_actor(self.agent, states)

        self._init_buffers()
        self._init_with_observation(self.env.reset())

    def sample(self, exploration_strategy=None):
        reward, num_steps, done_t = 0, 0, False

        while not done_t and self._sample_flag.value:
            state_t = self.get_state()
            action_t = self.policy_handler.action_fn(
                agent=self.agent,
                state=state_t,
                device=self._device,
                exploration_strategy=exploration_strategy,
                deterministic=self.deterministic
            )

            observation_tp1, reward_t, done_t, info = self.env.step(action_t)
            reward += reward_t

            transition = [observation_tp1, action_t, reward_t, done_t]
            self._put_transition(transition)
            num_steps += 1

        if not self._sample_flag.value:
            return None, None

        trajectory = self.get_trajectory()
        trajectory_info = {"reward": reward, "num_steps": num_steps}

        return trajectory, trajectory_info
