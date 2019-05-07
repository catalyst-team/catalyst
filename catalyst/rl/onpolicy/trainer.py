import os
import gc
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import multiprocessing as mp
import queue
import torch
from torch.utils.data import DataLoader

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.utils import \
    ReplayBufferDataset2, ReplayBufferSampler2
from catalyst.rl.db.core import DBSpec
from catalyst.rl.environments.core import EnvironmentSpec
from catalyst.rl.offpolicy.algorithms.core import AlgorithmSpec
from catalyst.rl.utils import _make_tuple, db2queue_loop


def get_states_from_observations(observations, history_len=1):
    """
    DB stores observations but not states.
    This function creates states from observations
    by adding new dimension of size (history_len).
    """
    observations = np.array(observations)
    episode_size = observations.shape[0]
    states = np.zeros(
        (episode_size, history_len) + observations.shape[1:])
    for i in range(history_len - 1):
        pivot = history_len - i - 1
        states[pivot:, i, :] = observations[:-pivot, :]
    states[:, -1, :] = observations
    return states


class Trainer:
    def __init__(
        self,
        algorithm: AlgorithmSpec,
        env_spec: EnvironmentSpec,
        db_server: DBSpec,
        logdir: str,
        num_workers: int = 1,
        batch_size: int = 64,
        epoch_len: int = int(1e2),
        num_mini_epochs: int = 10,
        min_num_trajectories: int = 100,
        min_num_transitions: int = 8192,
        max_num_transitions: int = None,
        save_period: int = 10,
        target_update_period: int = 1,
        online_update_period: int = 1,
        resume: str = None,
        gc_period: int = 10,
    ):
        # algorithm
        self.algorithm = algorithm
        if resume is not None:
            self.algorithm.load_checkpoint(resume)
        self.env_spec = env_spec

        # logging
        self.logdir = logdir
        current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
        logpath = f"{logdir}/trainer-{current_date}"
        os.makedirs(logpath, exist_ok=True)
        self.logger = SummaryWriter(logpath)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = 0
        self.epoch_len = epoch_len
        self.num_mini_epochs = num_mini_epochs

        # updates configuration
        # (actor_period, critic_period)
        self.actor_updates = 0
        self.critic_updates = 0

        # (actor_period, critic_period)
        self.actor_grad_period, self.critic_grad_period = \
            _make_tuple(online_update_period)

        # synchronization configuration
        self.db_server = db_server
        self.min_num_trajectories = min_num_trajectories
        self.min_num_transitions = min_num_transitions
        self.max_num_transitions = \
            max_num_transitions or (min_num_transitions * 2)

        self.save_period = save_period

        self.episodes_queue = mp.Queue()
        # self.episodes_queue = None
        self._db_loop_process = None
        self._num_trajectories = 0
        self._num_transitions = 0

        self._sampler_weight_mode = "actor"
        self._gc_period = gc_period

        self.replay_buffer = ReplayBufferDataset2(
            state_space=self.env_spec.state_space,
            action_space=self.env_spec.action_space,
            capacity=self.max_num_transitions,
            history_len=self.env_spec.history_len,
            n_step=self.algorithm.n_step,
            gamma=self.algorithm.gamma,
        )

    def save(self):
        if self.epoch % self.save_period == 0:
            checkpoint = self.algorithm.pack_checkpoint()
            checkpoint["epoch"] = self.epoch
            filename = UtilsFactory.save_checkpoint(
                logdir=self.logdir,
                checkpoint=checkpoint,
                suffix=str(self.epoch)
            )
            print("Checkpoint saved to: %s" % filename)

    def get_processes(self):
        processes = []
        if self._db_loop_process is not None:
            processes.append(self._db_loop_process)

        return processes

    def _start_db_loop(self):
        self._db_loop_process = mp.Process(
            target=db2queue_loop,
            kwargs={
                "db_server": self.db_server,
                "queue": self.episodes_queue,
                "max_size": int(1e3)
            }
        )
        self._db_loop_process.start()

    def _fetch_episodes(self):
        start_time = time.time()

        while self._num_trajectories < self.min_num_trajectories \
                and self._num_transitions < self.min_num_transitions:
            print(
                f"waiting for transitions, "
                f"{self._num_transitions:09d}\t"
                f"{self._num_trajectories:09d}")
            try:
                episode = self.episodes_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                time.sleep(1.0)
                continue
            self._num_trajectories += 1
            self._num_transitions += len(episode[-1])

            observations, actions, rewards, dones = episode
            states = get_states_from_observations(
                observations, self.env_spec.history_len)
            returns, values, advantages, action_logprobs = \
                self.algorithm.evaluate_trajectory(states, actions, rewards)
            episode = (
                states, actions, returns, values, advantages, action_logprobs)
            self.replay_buffer.push_episode(episode)

        elapsed_time = time.time() - start_time
        self.logger.add_scalar("fetch time", elapsed_time, self.epoch)

    def _update_samplers_weights(self):
        mode = self._sampler_weight_mode
        state_dict = self.algorithm.__dict__[mode].state_dict()
        state_dict = {
            k: v.detach().cpu().numpy()
            for k, v in state_dict.items()
        }
        self.db_server.dump_weights(weights=state_dict, prefix=mode)

    def _train(self):
        start_time = time.time()

        self.replay_buffer.rescale_advantages()

        sampler = ReplayBufferSampler2(
            buffer=self.replay_buffer,
            num_mini_epochs=self.num_mini_epochs)
        loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler)

        for i, batch in enumerate(loader):
            step_index = self.epoch * self.epoch_len + i + 1
            metrics = self.algorithm.train(
                batch,
                actor_update=(step_index % self.actor_grad_period == 0),
                critic_update=(step_index % self.critic_grad_period == 0)
            )

            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.add_scalar(key, value, step_index)

        elapsed_time = time.time() - start_time

        self.logger.add_scalar("batch size", self.batch_size, self.epoch)
        self.logger.add_scalar(
            "buffer size", len(self.replay_buffer), self.epoch)
        self.logger.add_scalar(
            "batches per second", i / elapsed_time, self.epoch
        )
        self.logger.add_scalar(
            "updates per second", i * self.batch_size / elapsed_time,
            self.epoch
        )

        self.save()
        self._update_samplers_weights()
        self.epoch += 1
        if self.epoch % self._gc_period == 0:
            gc.collect()

    def _start_train_loop(self):
        while True:
            if self._num_trajectories < self.min_num_trajectories \
                    and self._num_transitions < self.min_num_transitions:
                self.db_server.set_sample_flag(sample=True)
                self._fetch_episodes()
            else:
                # stop samplers
                self.db_server.set_sample_flag(sample=False)
                # train & update
                self._train()
                self._update_samplers_weights()
                # cleanup
                self.db_server.clean_trajectories()
                while not self.episodes_queue.empty():
                    self.episodes_queue.get()
                self._num_trajectories = 0
                self._num_transitions = 0

    def run(self):
        self._start_db_loop()
        self._update_samplers_weights()
        self._start_train_loop()
