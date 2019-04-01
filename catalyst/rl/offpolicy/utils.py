import numpy as np
import multiprocessing as mp
from gym.spaces import Discrete

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from catalyst.rl.environments.gym_wrapper import GymWrapper
from catalyst.rl.offpolicy.exploration.strategies import ParameterSpaceNoise


class ReplayBufferDataset(Dataset):
    def __init__(
        self,
        observation_shape,
        action_shape,
        max_size=int(1e6),
        n_step=1,
        gamma=0.99,
        history_len=1,
        discrete_actions=False,
        byte_observations=False
    ):
        """
        Experience replay buffer for off-policy RL algorithms.

        Args:
            observation_shape: shape of environment observation
                e.g. (8, ) for vector of floats or (84, 84, 3) for RGB image
            action_shape: shape of action the agent can take
                e.g. (3, ) for 3-dimensional continuous control
            max_size: replay buffer capacity
            history_len: number of subsequent observations considered a state
            n_step: number of time steps between the current state and the next
                state in TD backup
            gamma: discount factor
            discrete actions: True if actions are discrete
            byte_observations: True if observation values are ints in [0, 255]
                e.g. observations are RGB images
        """
        # @TODO: Refactor !!!
        self.observation_shape = observation_shape
        self.action_shape = (1, ) if discrete_actions else action_shape
        self.history_len = history_len
        self.n_step = n_step
        self.gamma = gamma
        self.max_size = max_size
        self.obs_dtype = np.unit8 if byte_observations else np.float32
        self.act_dtype = np.int if discrete_actions else np.float32
        self.len = 0
        self.pointer = 0

        self._store_lock = mp.RLock()

        self.observations = np.empty(
            (self.max_size, ) + self.observation_shape, dtype=self.obs_dtype
        )
        self.actions = np.empty(
            (self.max_size, ) + self.action_shape, dtype=self.act_dtype
        )
        self.rewards = np.empty((self.max_size, ), dtype=np.float32)
        self.dones = np.empty((self.max_size, ), dtype=np.bool)

    def push_episode(self, episode):
        with self._store_lock:
            observations, actions, rewards, dones = episode
            episode_len = len(rewards)
            self.len = min(self.len + episode_len, self.max_size)

            indices = np.arange(
                self.pointer, self.pointer + episode_len
            ) % self.max_size
            self.observations[indices] = np.array(observations)
            self.actions[indices] = np.array(actions)
            self.rewards[indices] = np.array(rewards)
            self.dones[indices] = np.array(dones)

            self.pointer = (self.pointer + episode_len) % self.max_size

    def get_state(self, idx, history_len=1):
        """ compose the state from a number (history_len) of observations
        """
        start_idx = idx - history_len + 1

        if start_idx < 0 or np.any(self.dones[start_idx:idx + 1]):
            state = np.zeros(
                (history_len, ) + self.observation_shape, dtype=self.obs_dtype
            )
            indices = [idx]
            for i in range(history_len - 1):
                next_idx = (idx - i - 1) % self.max_size
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
            (idx + n_step) % self.max_size, history_len
        )
        cum_reward = 0
        indices = np.arange(idx, idx + n_step) % self.max_size
        for num, i in enumerate(indices):
            cum_reward += self.rewards[i] * (gamma**num)
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


class ActionHandler:
    def __init__(
        self,
        env,
        actor,
        deterministic=False,
    ):
        self.env = env
        self.actor = actor
        self.deterministic = deterministic
        self._init()

    def _init(self):
        discrete_actions = isinstance(self.env.action_space, Discrete)

        # DQN
        if discrete_actions:
            if critic_distribution == "categorical":
                v_min, v_max = values_range
                z = torch.linspace(start=v_min, end=v_max, steps=n_atoms)
                self.z = self._to_tensor(z)
                self._act_fn = self._sample_from_categorical_critic
            elif critic_distribution == "quantile":
                self._act_fn = self._sample_from_quantile_critic
            else:
                self._act_fn = self._sample_from_critic
        # DDPG
        else:
            self.action_clip = \
                self.env.action_space.low, self.env.action_space.high
            self._act_fn = self._sample_from_actor

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def _sample_from_actor(self, actor, state):
        with torch.no_grad():
            states = self._to_tensor(state).unsqueeze(0)
            action = actor(states, deterministic=self.deterministic)
            action = action[0].detach().cpu().numpy()

        if self.action_clip is not None:
            action = np.clip(
                action,
                a_min=self.action_clip[0],
                a_max=self.action_clip[1]
            )
        return action

    def _sample_from_critic(self, critic, state):
        with torch.no_grad():
            states = self._to_tensor(state).unsqueeze(0)
            q_values = critic(states)[0]
            action = np.argmax(q_values.detach().cpu().numpy())
            return action

    def _sample_from_categorical_critic(self, critic, state):
        with torch.no_grad():
            states = self._to_tensor(state).unsqueeze(0)
            probs = F.softmax(critic(states)[0], dim=-1)
            q_values = torch.sum(probs * self.z, dim=-1)
            action = np.argmax(q_values.detach().cpu().numpy())
            return action

    def _sample_from_quantile_critic(self, critic, state):
        with torch.no_grad():
            states = self._to_tensor(state).unsqueeze(0)
            q_values = torch.mean(critic(states)[0], dim=-1)
            action = np.argmax(q_values.detach().cpu().numpy())
            return action

    def act(self, network, state, exploration_strategy=None):
        action = self._act_fn(network, state)

        if exploration_strategy is not None:
            action = exploration_strategy.update_action(action)

        return action


class EnvWrapper:
    def __init__(
        self,
        env: GymWrapper,
        actor,
        capacity,
        deterministic=False,
    ):
        self.env = env
        self.actor = actor
        self.capacity = capacity
        self.action_handler = ActionHandler(
            self.env, self.actor, deterministic=deterministic)

        self._init_buffers()

    def _init_buffers(self):
        self.pointer = 0
        self.observations = np.empty(
            (self.capacity,) + tuple(self.env.observation_space.shape),
            dtype=self.env.observation_space.dtype
        )
        self.actions = np.empty(
            (self.capacity,) + tuple(self.env.action_space.shape),
            dtype=self.env.action_space.dtype
        )
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.bool)

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self.actor.device)

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

    def _get_states_history(self, history_len=1):
        states = [
            self.get_state(history_len=history_len, pointer=i)
            for i in range(self.pointer)
        ]
        states = np.array(states)
        return states

    def get_state(self, pointer=None, history_len=None):
        pointer = pointer if pointer is not None else self.pointer
        history_len = history_len or self.env._history_len

        state = np.zeros(
            (history_len, ) + self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )

        indices = np.arange(max(0, pointer - history_len + 1), pointer + 1)
        state[-len(indices):] = self.observations[indices]
        return state

    def get_trajectory(self, tolist=False):
        indices = np.arange(self.pointer)
        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        if not tolist:
            trajectory = (observations, actions, rewards, dones)
        else:
            trajectory = (
                observations.tolist(),
                actions.tolist(),
                rewards.tolist(),
                dones.tolist()
            )
        return trajectory

    def reset(self, exploration_strategy=None):

        if isinstance(exploration_strategy, ParameterSpaceNoise) \
                and self.pointer > 1:
            with torch.no_grad():
                states = self._get_states_history()
                states = self._to_tensor(states)
                exploration_strategy.update_actor(self.actor, states)

        self._init_buffers()
        self._init_with_observation(self.env.reset())

    def play_episode(self, exploration_strategy):
        episode_reward, num_steps, done = 0, 0, False

        while not done:
            state = self.get_state()
            action = self.action_handler.act(state, exploration_strategy)

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
