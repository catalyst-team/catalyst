from typing import Dict
import time
import threading
import numpy as np

import torch
from torch.utils.data import DataLoader

from catalyst.rl.core import TrainerSpec, DBSpec
from catalyst.rl import utils


def _db2buffer_loop(db_server: DBSpec, buffer: utils.OffpolicyReplayBuffer):
    trajectory = None
    while True:
        if trajectory is None:
            trajectory = db_server.get_trajectory()

        if trajectory is not None:
            if buffer.push_trajectory(trajectory):
                trajectory = None
            else:
                time.sleep(1.0)
        else:
            if not db_server.training_enabled:
                return
            time.sleep(1.0)


class Trainer(TrainerSpec):
    def _init(
        self,
        target_update_period: int = 1,
        replay_buffer_size: int = int(1e6),
        replay_buffer_mode: str = "numpy",
        epoch_len: int = int(1e2),
        max_updates_per_sample: int = None,
        min_transitions_per_epoch: int = None,
    ):
        super()._init()
        # updates configuration
        # (actor_period, critic_period)
        self.actor_update_period, self.critic_update_period = \
            utils.make_tuple(target_update_period)
        self.actor_updates = 0
        self.critic_updates = 0

        #
        self.epoch_len = epoch_len
        self.max_updates_per_sample = max_updates_per_sample or np.inf
        self.min_transitions_per_epoch = min_transitions_per_epoch or -np.inf
        self.last_epoch_transitions = 0

        self.replay_buffer = utils.OffpolicyReplayBuffer(
            observation_space=self.env_spec.observation_space,
            action_space=self.env_spec.action_space,
            capacity=replay_buffer_size,
            history_len=self.env_spec.history_len,
            n_step=self.algorithm.n_step,
            gamma=self.algorithm.gamma,
            mode=replay_buffer_mode,
            logdir=self.logdir
        )

        self.replay_sampler = utils.OffpolicyReplaySampler(
            buffer=self.replay_buffer,
            epoch_len=self.epoch_len,
            batch_size=self.batch_size
        )

        self.loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=self.replay_sampler
        )

        self._db_loop_thread = None

    def _start_db_loop(self):
        self._db_loop_thread = threading.Thread(
            target=_db2buffer_loop,
            kwargs={
                "db_server": self.db_server,
                "buffer": self.replay_buffer,
            }
        )
        self._db_loop_thread.start()

    def _update_target_weights(self, update_step) -> Dict:
        output = {}

        if not self.env_spec.discrete_actions:
            if update_step % self.actor_update_period == 0:
                self.algorithm.target_actor_update()
                self.actor_updates += 1
                output["num_actor_updates"] = self.actor_updates

        if update_step % self.critic_update_period == 0:
            self.algorithm.target_critic_update()
            self.critic_updates += 1
            output["num_critic_updates"] = self.critic_updates

        return output

    def _run_epoch(self) -> Dict:
        self.replay_buffer.recalculate_index()

        expected_num_updates = (
            self.num_updates + len(self.loader) * self.loader.batch_size
        )
        expected_updates_per_sample = (
            expected_num_updates / self.replay_buffer.num_transitions
        )
        min_epoch_transitions = \
            self.last_epoch_transitions + self.min_transitions_per_epoch
        while expected_updates_per_sample > self.max_updates_per_sample \
                or self.replay_buffer.num_transitions < min_epoch_transitions:
            time.sleep(5.0)
            self.replay_buffer.recalculate_index()
            expected_num_updates = (
                self.num_updates + len(self.loader) * self.loader.batch_size
            )
            expected_updates_per_sample = (
                expected_num_updates / self.replay_buffer.num_transitions
            )

        metrics = self._run_loader(self.loader)
        self.last_epoch_transitions = self.replay_buffer.num_transitions

        updates_per_sample = (
            self.num_updates / self.replay_buffer.num_transitions
        )
        metrics.update(
            {
                "num_trajectories": self.replay_buffer.num_trajectories,
                "num_transitions": self.replay_buffer.num_transitions,
                "buffer_size": len(self.replay_buffer),
                "updates_per_sample": updates_per_sample
            }
        )
        return metrics

    def _fetch_initial_buffer(self):
        buffer_size = len(self.replay_buffer)
        while buffer_size < self.min_num_transitions:
            self.replay_buffer.recalculate_index()

            num_trajectories = self.replay_buffer.num_trajectories
            num_transitions = self.replay_buffer.num_transitions
            buffer_size = len(self.replay_buffer)

            metrics = [
                f"fps: {0:7.1f}",
                f"updates per sample: {0:7.1f}",
                f"trajectories: {num_trajectories:09d}",
                f"transitions: {num_transitions:09d}",
                f"buffer size: "
                f"{buffer_size:09d}/{self.min_num_transitions:09d}",
            ]
            metrics = " | ".join(metrics)
            print(f"--- {metrics}")

            time.sleep(1.0)

    def _start_train_loop(self):
        self._start_db_loop()
        self.db_server.push_message(self.db_server.Message.ENABLE_TRAINING)
        self.db_server.push_message(self.db_server.Message.ENABLE_SAMPLING)
        self._fetch_initial_buffer()
        self._run_train_loop()
