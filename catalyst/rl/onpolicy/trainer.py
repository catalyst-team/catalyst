from typing import Dict

import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from catalyst.rl.core import TrainerSpec
from catalyst.rl import utils


def _get_states_from_observations(observations: np.ndarray, history_len=1):
    """
    DB stores observations but not states.
    This function creates states from observations
    by adding new dimension of size (history_len).
    """
    states = np.concatenate(
        [np.expand_dims(np.zeros_like(observations), 1)] * history_len,
        axis=1)
    for i in range(history_len - 1):
        pivot = history_len - i - 1
        states[pivot:, i, ...] = observations[:-pivot, ...]
    states[:, -1, ...] = observations

    # structed numpy array
    if observations.dtype.fields is not None:
        states_dtype = []
        for key, value in observations.dtype.fields.items():
            states_dtype.append(
                (key, value[0].base, (history_len,) + tuple(value[0].shape)))
        states_dtype = np.dtype(states_dtype)
        states_ = np.empty(len(observations), dtype=states_dtype)

        for key in observations.dtype.fields.keys():
            states_[key] = states[key]

        states = states_

    return states


class Trainer(TrainerSpec):
    def _init(
        self,
        num_mini_epochs: int = 10,
        min_num_trajectories: int = 100,
        rollout_batch_size: int = None
    ):
        self.num_mini_epochs = num_mini_epochs
        self.min_num_trajectories = min_num_trajectories
        self.max_num_transitions = self.min_num_transitions * 3
        self.rollout_batch_size = rollout_batch_size

    def _get_rollout_in_batches(self, states, actions, rewards, dones):

        if self.rollout_batch_size is None:
            return self.algorithm.get_rollout(states, actions, rewards, dones)

        indices = np.arange(
            0, len(states) + self.rollout_batch_size - 1,
            self.rollout_batch_size
        )
        rollout = None
        for i in range(len(indices) - 1):
            states_batch = states[indices[i]:indices[i+1]+1]
            actions_batch = actions[indices[i]:indices[i+1]+1]
            rewards_batch = rewards[indices[i]:indices[i+1]+1]
            dones_batch = dones[indices[i]:indices[i+1]+1]
            rollout_batch = self.algorithm.get_rollout(
                states_batch, actions_batch, rewards_batch, dones_batch
            )
            if rollout is not None:
                rollout = utils.append_dict(rollout, rollout_batch)
            else:
                rollout = rollout_batch
        return rollout

    def _fetch_trajectories(self):

        # cleanup trajectories
        self.db_server.clean_trajectories()
        num_trajectories = 0
        num_transitions = 0
        del self.replay_buffer

        rollout_spec = self.algorithm.get_rollout_spec()
        self.replay_buffer = utils.OnpolicyRolloutBuffer(
            state_space=self.env_spec.state_space,
            action_space=self.env_spec.action_space,
            capacity=self.max_num_transitions,
            **rollout_spec
        )

        # start samplers
        self.db_server.set_sample_flag(sample=True)

        start_time = time.time()

        while num_trajectories < self.min_num_trajectories \
                and num_transitions < self.min_num_transitions:

            trajectories_percentrage = \
                100 * num_trajectories / self.min_num_trajectories
            trajectories_stats = \
                f"{num_trajectories:09d} / " \
                f"{self.min_num_trajectories:09d} " \
                f"({trajectories_percentrage:5.2f}%)"
            transitions_percentrage = \
                100 * num_transitions / self.min_num_transitions
            transitions_stats = \
                f"{num_transitions:09d} / " \
                f"{self.min_num_transitions:09d} " \
                f"({transitions_percentrage:5.2f}%)"
            print(
                f"trajectories: {trajectories_stats}\t"
                f"transitions: {transitions_stats}\t"
            )

            try:
                trajectory = self.db_server.get_trajectory()
                assert trajectory is not None
            except AssertionError:
                time.sleep(1.0)
                continue

            num_trajectories += 1
            num_transitions += len(trajectory[-1])

            observations, actions, rewards, dones = trajectory
            states = _get_states_from_observations(
                observations, self.env_spec.history_len)
            rollout = self._get_rollout_in_batches(
                states, actions, rewards, dones
            )
            self.replay_buffer.push_rollout(
                state=states,
                action=actions,
                reward=rewards,
                **rollout,
            )

        # stop samplers
        self.db_server.set_sample_flag(sample=False)

        self._num_trajectories += num_trajectories
        self._num_transitions += num_transitions

        # @TODO: refactor
        self.algorithm.postprocess_buffer(
            self.replay_buffer.buffers,
            len(self.replay_buffer))

        elapsed_time = time.time() - start_time
        self.logger.add_scalar("fetch time", elapsed_time, self.epoch)

    def _run_epoch(self) -> Dict:
        sampler = utils.OnpolicyRolloutSampler(
            buffer=self.replay_buffer,
            num_mini_epochs=self.num_mini_epochs)
        loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler)

        metrics = self._run_loader(loader)

        updates_per_sample = self.num_updates / self._num_transitions
        metrics.update({
            "num_trajectories": self._num_trajectories,
            "num_transitions": self._num_transitions,
            "buffer_size": len(self.replay_buffer),
            "updates_per_sample": updates_per_sample
        })
        return metrics

    def _run_train_loop(self):
        while True:
            # get trajectories
            self._fetch_trajectories()
            # train & update
            self._run_epoch_loop()
