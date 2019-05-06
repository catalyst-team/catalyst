from typing import Union
import numpy as np
import multiprocessing as mp
from gym.spaces import Box, Discrete, Space

import torch
from torch.utils.data import Dataset, Sampler
from catalyst.rl.offpolicy.exploration.strategies import \
    OrnsteinUhlenbeckProcess, ParameterSpaceNoise
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


def _get_buffers(
    observation_space: Space,
    action_space: Space,
    capacity: int,
    mode: str = "numpy",
    logdir: str = None
):
    assert mode in ["numpy", "memmap"]
    if mode == "numpy":
        observations = np.empty(
            (capacity,) + tuple(observation_space.shape),
            dtype=observation_space.dtype
        )
        actions = np.empty(
            (capacity,) + tuple(action_space.shape),
            dtype=action_space.dtype
        )
        rewards = np.empty((capacity,), dtype=np.float32)
        dones = np.empty((capacity,), dtype=np.bool)
    elif mode == "memmap":
        assert logdir is not None

        observations = np.memmap(
            f"{logdir}/observations.memmap",
            mode="w+",
            shape=(capacity,) + tuple(observation_space.shape),
            dtype=observation_space.dtype
        )
        actions = np.memmap(
            f"{logdir}/actions.memmap",
            mode="w+",
            shape=(capacity,) + tuple(action_space.shape),
            dtype=action_space.dtype
        )
        rewards = np.memmap(
            f"{logdir}/rewards.memmap",
            mode="w+",
            shape=(capacity,),
            dtype=np.float32
        )
        dones = np.memmap(
            f"{logdir}/dones.memmap",
            mode="w+",
            shape=(capacity,),
            dtype=np.bool
        )
    else:
        raise NotImplementedError()

    return observations, actions, rewards, dones


class ReplayBufferDataset(Dataset):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        capacity=int(1e6),
        n_step=1,
        gamma=0.99,
        history_len=1,
        mode="numpy",
        logdir=None
    ):
        """
        Experience replay buffer for off-policy RL algorithms.

        Args:
            observation_shape: shape of environment observation
                e.g. (8, ) for vector of floats or (84, 84, 3) for RGB image
            action_shape: shape of action the agent can take
                e.g. (3, ) for 3-dimensional continuous control
            capacity: replay buffer capacity
            history_len: number of subsequent observations considered a state
            n_step: number of time steps between the current state and the next
                state in TD backup
            gamma: discount factor
            discrete actions: True if actions are discrete
        """
        # @TODO: Refactor !!!
        self.observation_space = observation_space
        self.action_space = action_space
        self.history_len = history_len

        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        self.len = 0
        self.pointer = 0
        self._store_lock = mp.RLock()

        self.observations, self.actions, self.rewards, self.dones = \
            _get_buffers(
                capacity=capacity,
                observation_space=observation_space,
                action_space=action_space,
                mode=mode,
                logdir=logdir
            )

    def push_episode(self, episode):
        with self._store_lock:
            observations, actions, rewards, dones = episode
            episode_len = len(rewards)
            self.len = min(self.len + episode_len, self.capacity)

            indices = np.arange(
                self.pointer, self.pointer + episode_len
            ) % self.capacity
            self.observations[indices] = np.array(observations)
            self.actions[indices] = np.array(actions)
            self.rewards[indices] = np.array(rewards)
            self.dones[indices] = np.array(dones)

            self.pointer = (self.pointer + episode_len) % self.capacity

    def get_state(self, idx, history_len=1):
        """ compose the state from a number (history_len) of observations
        """
        start_idx = idx - history_len + 1

        if start_idx < 0 or np.any(self.dones[start_idx:idx + 1]):
            state = np.zeros(
                (history_len,) + self.observation_space.shape,
                dtype=self.observation_space.dtype
            )
            indices = [idx]
            for i in range(history_len - 1):
                next_idx = (idx - i - 1) % self.capacity
                if next_idx >= self.len or self.dones[next_idx]:
                    break
                indices.append(next_idx)
            indices = indices[::-1]
            state[-len(indices):] = self.observations[indices]
        else:
            state = self.observations[slice(start_idx, idx + 1, 1)]

        return state

    def get_transition_n_step(self, idx, history_len=1, n_step=1, gamma=0.99):
        state = self.get_state(idx, history_len)
        next_state = self.get_state(
            (idx + n_step) % self.capacity, history_len
        )
        cum_reward = 0
        indices = np.arange(idx, idx + n_step) % self.capacity
        for num, i in enumerate(indices):
            cum_reward += self.rewards[i] * (gamma ** num)
            done = self.dones[i]
            if done:
                break
        return state, self.actions[idx], cum_reward, next_state, done

    def __getitem__(self, index):
        with self._store_lock:
            state, action, reward, next_state, done = \
                self.get_transition_n_step(
                    index,
                    history_len=self.history_len,
                    n_step=self.n_step,
                    gamma=self.gamma)

        dct = {
            "state": np.array(state).astype(np.float32),
            "action": np.array(action).astype(np.float32),
            "reward": np.array(reward).astype(np.float32),
            "next_state": np.array(next_state).astype(np.float32),
            "done": np.array(done).astype(np.float32)
        }

        return dct

    def __len__(self):
        return self.len


class ReplayBufferSampler(Sampler):
    def __init__(self, buffer, epoch_len, batch_size):
        super().__init__(None)
        self.buffer = buffer
        self.buffer_history_len = buffer.history_len
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.len = self.epoch_len * self.batch_size

    def __iter__(self):
        indices = np.random.choice(range(len(self.buffer)), size=self.len)
        return iter(indices)

    def __len__(self):
        return self.len


class PolicyHandler:
    def __init__(
        self,
        env: EnvironmentSpec,
        agent: Union[ActorSpec, CriticSpec],
        device
    ):
        self.discrete_actions = isinstance(env.action_space, Discrete)

        # DQN
        if self.discrete_actions:
            assert isinstance(agent, CriticSpec)
            self.value_distribution = agent.distribution
            if self.value_distribution == "categorical":
                v_min, v_max = agent.values_range
                self.z = torch.linspace(
                    start=v_min,
                    end=v_max,
                    steps=agent.num_atoms
                ).to(device)

        # DDPG
        else:
            assert isinstance(agent, ActorSpec)
            action_space: Box = env.action_space
            self.action_clip = action_space.low, action_space.high

    @torch.no_grad()
    def _get_q_values(
        self,
        critic: CriticSpec,
        state: np.ndarray,
        device
    ):
        states = torch.Tensor(state).to(device).unsqueeze(0)
        if self.value_distribution == "categorical":
            probs = torch.softmax(critic(states)[0], dim=-1)
            q_values = torch.sum(probs * self.z, dim=-1)
        elif self.value_distribution == "quantile":
            q_values = torch.mean(critic(states)[0], dim=-1)
        else:
            q_values = critic(states)[0]
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
                action,
                a_min=self.action_clip[0],
                a_max=self.action_clip[1]
            )
        return action


class EpisodeRunner:
    def __init__(
        self,
        env: EnvironmentSpec,
        agent: Union[ActorSpec, CriticSpec],
        device,
        capacity: int,
        deterministic: bool = False,
    ):
        self.env = env
        self.agent = agent
        self._device = device
        self.capacity = capacity
        self.deterministic = deterministic
        self.policy_handler = PolicyHandler(
            env=self.env, agent=self.agent, device=device
        )

        if isinstance(env.action_space, Discrete):
            self._act_fn = self._act_discrete
        else:
            self._act_fn = self._act_continuous

        self._init_buffers()

    def _init_buffers(self):
        self.pointer = 0
        self.observations, self.actions, self.rewards, self.dones = \
            _get_buffers(
                capacity=self.capacity,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def _init_with_observation(self, observation):
        self.observations[0] = observation
        self.pointer = 0

    def _put_transition(self, transition):
        """
        transition = [s_tp1, a_t, r_t, d_t]
        """
        s_tp1, a_t, r_t, d_t = transition
        self.observations[self.pointer + 1] = s_tp1
        self.actions[self.pointer] = a_t
        self.rewards[self.pointer] = r_t
        self.dones[self.pointer] = d_t
        self.pointer += 1

    def _get_states_history(self, history_len=None):
        history_len = history_len or self.env.history_len
        states = [
            self.get_state(history_len=history_len, pointer=i)
            for i in range(self.pointer)
        ]
        states = np.array(states)
        return states

    def get_state(self, pointer=None, history_len=None):
        pointer = pointer if pointer is not None else self.pointer
        history_len = history_len or self.env.history_len

        state = np.zeros(
            (history_len,) + self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )

        indices = np.arange(max(0, pointer - history_len + 1), pointer + 1)
        state[-len(indices):] = self.observations[indices]
        return state

    def get_trajectory(self):
        indices = np.arange(self.pointer)

        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        trajectory = (observations, actions, rewards, dones)
        return trajectory

    @torch.no_grad()
    def reset(self, exploration_strategy=None):

        if isinstance(exploration_strategy, OrnsteinUhlenbeckProcess):
            exploration_strategy.reset_state(self.env.action_space.shape[0])

        if isinstance(exploration_strategy, ParameterSpaceNoise) \
                and self.pointer > 1:
            states = self._get_states_history()
            states = self._to_tensor(states)
            exploration_strategy.update_actor(self.agent, states)

        self._init_buffers()
        self._init_with_observation(self.env.reset())

    def play_episode(self, exploration_strategy):
        episode_reward, num_steps, done = 0, 0, False

        while not done:
            state = self.get_state()
            action = self._act_fn(
                agent=self.agent,
                state=state,
                device=self._device,
                exploration_strategy=exploration_strategy,
                deterministic=self.deterministic
            )

            next_observation, reward, done, info = self.env.step(action)
            episode_reward += reward

            transition = [next_observation, action, reward, done]
            self._put_transition(transition)
            num_steps += 1

        results = {
            "episode_reward": episode_reward,
            "num_steps": num_steps
        }

        return results

    def _act_discrete(
        self,
        agent: CriticSpec,
        state: np.ndarray,
        device,
        deterministic: bool = False,
        exploration_strategy=None
    ):
        q_values = self.policy_handler._get_q_values(agent, state, device)
        action = exploration_strategy.get_action(q_values)
        return action

    def _act_continuous(
        self,
        agent: ActorSpec,
        state: np.ndarray,
        device,
        deterministic: bool = False,
        exploration_strategy=None
    ):
        action = self.policy_handler._sample_from_actor(
            agent, state, device, deterministic
        )
        action = exploration_strategy.get_action(action)
        return action
