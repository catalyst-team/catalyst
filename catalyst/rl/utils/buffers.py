from typing import Dict, Tuple

import numpy as np
import multiprocessing as mp
from gym import spaces
from torch.utils.data import Dataset


def _get_buffer(
    capacity: int,
    space: spaces.Space = None,
    shape: Tuple = None,
    dtype = None,
    name: str = None,
    mode: str = "numpy",
    logdir: str = None
):
    assert mode in ["numpy", "memmap"]
    assert \
        (space is None and shape is not None and dtype is not None) \
        or (space is not None and shape is None and dtype is None)

    if mode == "numpy":
        if space is None or not isinstance(space, spaces.Dict):
            space_shape = shape if space is None else space.shape
            space_dtype = dtype if space is None else space.dtype

            buffer_dtype = space_dtype
            buffer = np.empty(
                (capacity, ) + tuple(space_shape),
                dtype=space_dtype
            )
        else:
            assert space is not None

            buffer_dtype = []
            for key, value in space.spaces.items():
                buffer_dtype.append((key, value.dtype, value.shape))
            buffer_dtype = np.dtype(buffer_dtype)
            buffer = np.empty(capacity, dtype=buffer_dtype)
    elif mode == "memmap":
        assert logdir is not None
        assert name is not None

        if space is None or not isinstance(space, spaces.Dict):
            space_shape = shape if space is None else space.shape
            space_dtype = dtype if space is None else space.dtype

            buffer_dtype = space_dtype
            buffer = np.memmap(
                f"{logdir}/{name}.memmap",
                mode="w+",
                shape=(capacity,) + tuple(space_shape),
                dtype=space_dtype
            )
        else:
            assert space is not None

            buffer_dtype = []
            for key, value in space.spaces.items():
                buffer_dtype.append((key, value.dtype, value.shape))
            buffer_dtype = np.dtype(buffer_dtype)
            buffer = np.memmap(
                f"{logdir}/{name}.memmap",
                mode="w+",
                shape=(capacity,),
                dtype=buffer_dtype
            )
    else:
        raise NotImplementedError()

    return buffer, buffer_dtype


class OffpolicyReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        capacity: int = int(1e6),
        capacity_mult: int = 2,
        n_step: int = 1,
        gamma: float = 0.99,
        history_len: int = 1,
        mode: str = "numpy",
        logdir: str = None
    ):
        """
        Experience replay buffer for off-policy RL algorithms.

        Args:
            observation_space: space of environment observation
                e.g. (8, ) for vector of floats or (84, 84, 3) for RGB image
            action_space: space of action the agent can take
                e.g. (3, ) for 3-dimensional continuous control
            capacity: replay buffer capacity
            n_step: number of time steps between the current state and the next
                state in TD backup
            gamma: discount factor
            history_len: number of subsequent observations considered a state
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.history_len = history_len
        self.n_step = n_step
        self.gamma = gamma

        self.length = 0
        self.capacity = capacity
        self.capacity_mult = capacity_mult
        self.capacity_limit = capacity * capacity_mult

        self._store_lock = mp.Lock()
        self.num_trajectories = 0
        self.num_transitions = 0
        self.pointer = 0
        self._trajectories_lens = []

        self.observations, self.observations_dtype = _get_buffer(
            capacity=self.capacity_limit,
            space=self.observation_space,
            name="observations",
            mode=mode,
            logdir=logdir
        )
        self.actions, self.actions_dtype = _get_buffer(
            capacity=self.capacity_limit,
            space=self.action_space,
            name="actions",
            mode=mode,
            logdir=logdir
        )
        self.rewards, self.rewards_dtype = _get_buffer(
            capacity=self.capacity_limit,
            shape=(),
            dtype=np.float32,
            name="rewards",
            mode=mode,
            logdir=logdir
        )
        self.dones, self.dones_dtype = _get_buffer(
            capacity=self.capacity_limit,
            shape=(),
            dtype=np.bool,
            name="dones",
            mode=mode,
            logdir=logdir
        )

    def push_trajectory(self, trajectory):
        with self._store_lock:
            observations, actions, rewards, dones = trajectory
            trajectory_len = len(rewards)

            if self.pointer + trajectory_len >= self.capacity_limit:
                return False

            if isinstance(self.observations_dtype, np.dtype):
                observations_ = np.zeros(1, dtype=self.observations_dtype)
                for key in observations:
                    observations_[key] = observations[key]
            else:
                observations_ = np.array(observations)
            self.observations[self.pointer:self.pointer + trajectory_len] = \
                observations_

            if isinstance(self.actions_dtype, np.dtype):
                actions_ = np.zeros(1, dtype=self.actions_dtype)
                for key in actions:
                    actions_[key] = actions[key]
            else:
                actions_ = np.array(actions)
            self.actions[self.pointer:self.pointer + trajectory_len] = \
                actions_

            self.rewards[self.pointer:self.pointer + trajectory_len] = \
                np.array(rewards)
            self.dones[self.pointer:self.pointer + trajectory_len] = \
                np.array(dones)

            self._trajectories_lens.append(trajectory_len)
            self.pointer += trajectory_len
            self.num_trajectories += 1
            self.num_transitions += trajectory_len

        return True

    def recalculate_index(self):
        with self._store_lock:
            curr_p = self.pointer
            if curr_p > self.capacity:
                diff = curr_p - self.capacity

                tr_cumsum = np.cumsum(self._trajectories_lens)
                tr_cumsum_mask = tr_cumsum < diff
                tr_offset = np.where(tr_cumsum_mask, tr_cumsum, -1)
                offset = tr_offset.argmax()
                offset += 1 if tr_offset[offset] > -1 else 0

                self._trajectories_lens = self._trajectories_lens[offset + 1:]
                offset = tr_cumsum[offset]
                curr_p = curr_p - offset

                delta = int(1e5)
                for i in range(0, curr_p, delta):
                    i_start = i * delta
                    i_end = min((i + 1) * delta, curr_p)
                    self.observations[i_start:i_end] = \
                        self.observations[offset+i_start:offset+i_end]
                    self.actions[i_start:i_end] = \
                        self.actions[offset + i_start:offset + i_end]
                    self.rewards[i_start:i_end] = \
                        self.rewards[offset + i_start:offset + i_end]
                    self.dones[i_start:i_end] = \
                        self.dones[offset + i_start:offset + i_end]

                self.pointer = curr_p
            self.length = curr_p

    def get_state(self, idx, history_len=1):
        """
        compose the state from a number (history_len) of observations
        """
        start_idx = idx - history_len + 1

        if start_idx < 0 or np.any(self.dones[start_idx:idx + 1]):
            if isinstance(self.observations_dtype, np.dtype):
                state = np.zeros(
                    (history_len,),
                    dtype=self.observations_dtype
                )
            else:
                state = np.zeros(
                    (history_len, ) + self.observation_space.shape,
                    dtype=self.observation_space.dtype
                )
            indices = [idx]
            for i in range(history_len - 1):
                next_idx = (idx - i - 1) % self.capacity

                if next_idx >= self.length or self.dones[next_idx]:
                    break
                indices.append(next_idx)
            indices = indices[::-1]
            state[-len(indices):] = self.observations[indices]
        else:
            state = self.observations[slice(start_idx, idx + 1, 1)]

        return state

    def get_transition_n_step(self, idx, history_len=1, n_step=1, gamma=0.99):
        state = self.get_state(idx, history_len)
        next_state = self.get_state((idx + n_step) % self.length, history_len)
        cum_reward = 0
        indices = np.arange(idx, idx + n_step) % self.length
        for num, i in enumerate(indices):
            cum_reward += self.rewards[i] * (gamma**num)
            done = self.dones[i]
            if done:
                break
        return state, self.actions[idx], cum_reward, next_state, done

    def __getitem__(self, index):
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
        return self.length


class OnpolicyRolloutBuffer(Dataset):
    def __init__(
        self,
        state_space: spaces.Space,
        action_space: spaces.Space,
        capacity=int(1e6),
        **rollout_spec
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.capacity = capacity
        self.len = 0
        self.pointer = 0

        self.buffers = {
            "state": BufferWrapper(
                capacity=self.capacity,
                space=state_space,
            ),
            "action": BufferWrapper(
                capacity=self.capacity,
                space=action_space,
            )
        }

        for key, value in rollout_spec.items():
            self.buffers[key] = BufferWrapper(
                capacity=capacity,
                shape=value["shape"],
                dtype=value["dtype"]
            )

    def push_rollout(self, **rollout: Dict):
        trajectory_len = len(rollout["state"])
        self.len = min(self.len + trajectory_len, self.capacity)
        indices = np.arange(
            self.pointer, self.pointer + trajectory_len
        ) % self.capacity
        self.pointer = (self.pointer + trajectory_len) % self.capacity

        for key in self.buffers:
            self.buffers[key][indices] = rollout[key]

    def __getitem__(self, index):
        dct = {
            key: np.array(value[index]).astype(np.float32)
            for key, value in self.buffers.items()
        }
        return dct

    def __len__(self):
        return self.len
