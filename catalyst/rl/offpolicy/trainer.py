import os
import time
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import multiprocessing as mp
import queue
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from catalyst.dl.utils import UtilsFactory
from catalyst.utils.serialization import serialize, deserialize


class BufferDataset(Dataset):
    def __init__(
        self,
        state_shape,
        action_shape,
        max_size=int(1e6),
        history_len=1,
        n_step=1,
        gamma=0.99
    ):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.history_len = history_len
        self.n_step = n_step
        self.gamma = gamma
        self.max_size = max_size
        self.len = 0
        self.pointer = 0

        self._store_lock = mp.RLock()
        self.states = np.empty(
            (self.max_size, ) + self.state_shape, dtype=np.float32
        )
        self.actions = np.empty(
            (self.max_size, ) + self.action_shape, dtype=np.float32
        )
        self.rewards = np.empty((self.max_size, ), dtype=np.float32)
        self.dones = np.empty((self.max_size, ), dtype=np.bool)

    def push_episode(self, episode):
        with self._store_lock:
            states, actions, rewards, dones = episode
            episode_len = len(rewards)
            self.len = min(self.len + episode_len, self.max_size)

            indices = np.arange(
                self.pointer, self.pointer + episode_len
            ) % self.max_size
            self.states[indices] = np.array(states)
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
                (history_len, ) + self.state_shape, dtype=np.float32
            )
            indices = [idx]
            for i in range(history_len - 1):
                next_idx = (idx - i - 1) % self.max_size
                if next_idx >= self.len or self.dones[next_idx]:
                    break
                indices.append(next_idx)
            indices = indices[::-1]
            state[-len(indices):] = self.states[indices]
        else:
            state = self.states[slice(start_idx, idx + 1, 1)]

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


class BufferSampler(Sampler):
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


def redis2queue_loop(redis, queue, max_size):
    pointer = 0
    redis_len = redis.llen("trajectories") - 1
    while True:
        try:
            need_more = pointer < redis_len and queue.qsize() < max_size
        except NotImplementedError:  # MacOS qsize issue (no sem_getvalue)
            need_more = pointer < redis_len

        if need_more:
            episode = deserialize(redis.lindex("trajectories", pointer))
            queue.put(episode, block=True, timeout=1.0)
            pointer += 1
        else:
            time.sleep(1.0)

        redis_len = redis.llen("trajectories") - 1


class Trainer:
    def __init__(
        self,
        algorithm,
        state_shape,
        action_shape,
        logdir,
        redis_server=None,
        redis_prefix=None,
        num_workers=1,
        replay_buffer_size=int(1e6),
        batch_size=64,
        start_learning=int(1e3),
        gamma=0.99,
        n_step=1,
        history_len=1,
        epoch_len=int(1e2),
        save_period=10,
        target_update_period=1,
        online_update_period=1,
        weights_sync_period=1,
        max_redis_trials=1000
    ):

        self.algorithm = algorithm
        history_len = history_len

        self.logdir = logdir
        current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
        logpath = f"{logdir}/trainer-{current_date}"
        os.makedirs(logpath, exist_ok=True)
        self.logger = SummaryWriter(logpath)

        self.episodes_queue = mp.Queue()

        self.buffer = BufferDataset(
            state_shape=state_shape,
            action_shape=action_shape,
            max_size=replay_buffer_size,
            history_len=history_len,
            n_step=n_step,
            gamma=gamma
        )

        self.gamma = gamma
        self.n_step = n_step
        self.history_len = history_len

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sampler = BufferSampler(
            buffer=self.buffer, epoch_len=epoch_len, batch_size=batch_size
        )

        self.loader = DataLoader(
            dataset=self.buffer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=self.sampler
        )

        self.redis_server = redis_server
        self.redis_prefix = redis_prefix
        self.max_redis_trials = max_redis_trials
        self.start_learning = start_learning

        self.epoch = 0
        self.epoch_len = epoch_len

        # (actor_period, critic_period)
        target_update_period = (
            target_update_period if isinstance(target_update_period, list) else
            (target_update_period, target_update_period)
        )
        self.actor_update_period, self.critic_update_period = \
            target_update_period

        # (actor_period, critic_period)
        online_update_period = (
            online_update_period if isinstance(online_update_period, list) else
            (online_update_period, online_update_period)
        )
        self.actor_grad_period, self.critic_grad_period = \
            online_update_period

        self.save_period = save_period
        self.weights_sync_period = weights_sync_period

        self.actor_updates = 0
        self.critic_updates = 0

        self._redis_loop_process = None

    def __repr__(self):
        str_val = " ".join(
            [
                f"{key}: {str(getattr(self, key, ''))}"
                for key in ["algorithm", "n_step", "gamma", "history_len"]
            ]
        )
        return f"Trainer. {str_val}"

    def get_processes(self):
        processes = []
        if self._redis_loop_process is not None:
            processes.append(self._redis_loop_process)

        return processes

    def run(self):
        self.update_samplers_weights()
        self.start_redis_loop()
        self.start_train_loop()

    def start_redis_loop(self):
        self._redis_loop_process = mp.Process(
            target=redis2queue_loop,
            kwargs={
                "redis": self.redis_server,
                "queue": self.episodes_queue,
                "max_size": int(self.max_redis_trials * 2)
            }
        )
        self._redis_loop_process.start()

    def fetch_episodes(self):
        for i in range(self.max_redis_trials):
            try:
                episode = self.episodes_queue.get(block=True, timeout=1.0)
                self.buffer.push_episode(episode)
            except queue.Empty:
                break
        stored = len(self.buffer)
        print(f"transitions: {stored}")
        return i

    def start_train_loop(self):
        stored = len(self.buffer)
        while stored < self.start_learning:
            self.fetch_episodes()
            stored = len(self.buffer)
            print(f"waiting for transitions: {stored}/{self.start_learning}")
            time.sleep(1.0)
        self.train()

    def train(self):
        start_time = time.time()
        while True:
            fetch_i = self.fetch_episodes()
            elapsed_time = time.time() - start_time
            self.logger.add_scalar("fetch time", elapsed_time, self.epoch)
            self.logger.add_scalar("fetch index", fetch_i, self.epoch)
            start_time = time.time()

            for i, batch in enumerate(self.loader):
                step_index = self.epoch * self.epoch_len + i + 1
                metrics = self.algorithm.train(
                    batch,
                    actor_update=(step_index % self.actor_grad_period == 0),
                    critic_update=(step_index % self.critic_grad_period == 0)
                )

                for key, value in metrics.items():
                    if isinstance(value, float):
                        self.logger.add_scalar(key, value, step_index)

                self.update_target_weights(step_index)

            elapsed_time = time.time() - start_time
            self.logger.add_scalar("batch size", self.batch_size, self.epoch)
            self.logger.add_scalar("buffer size", len(self.buffer), self.epoch)
            self.logger.add_scalar(
                "batches per second", i / elapsed_time, self.epoch
            )
            self.logger.add_scalar(
                "updates per second", i * self.batch_size / elapsed_time,
                self.epoch
            )

            self.save()
            self.update_samplers_weights()
            self.epoch += 1
            start_time = time.time()

    def update_target_weights(self, step_index):
        if step_index % self.actor_update_period == 0:
            self.algorithm.target_actor_update()
            self.actor_updates += 1

            self.logger.add_scalar(
                "actor updates", self.actor_updates, step_index
            )

        if step_index % self.critic_update_period == 0:
            self.algorithm.target_critic_update()
            self.critic_updates += 1

            self.logger.add_scalar(
                "critic updates", self.critic_updates, step_index
            )

    def save(self):
        if self.epoch % self.save_period == 0:
            checkpoint = self.algorithm.prepare_checkpoint()
            checkpoint["epoch"] = self.epoch
            filename = UtilsFactory.save_checkpoint(
                logdir=self.logdir,
                checkpoint=checkpoint,
                suffix=str(self.epoch)
            )
            print("Checkpoint saved to: %s" % filename)

    def update_samplers_weights(self):
        if self.epoch % self.weights_sync_period == 0:
            actor_state_dict = self.algorithm.actor.state_dict()
            actor_state_dict = {
                k: v.tolist()
                for k, v in actor_state_dict.items()
            }
            self.redis_server.set(
                f"{self.redis_prefix}_actor_weights",
                serialize(actor_state_dict)
            )
