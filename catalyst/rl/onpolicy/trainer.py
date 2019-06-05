import time

import torch
from torch.utils.data import DataLoader

from catalyst.rl.core import TrainerSpec
from catalyst.rl.utils import \
    OnpolicyRolloutBuffer, OnpolicyRolloutSampler, \
    _get_states_from_observations


class Trainer(TrainerSpec):
    def _init(
        self,
        num_mini_epochs: int = 10,
        min_num_trajectories: int = 100,
    ):
        self.num_mini_epochs = num_mini_epochs
        self.min_num_trajectories = min_num_trajectories
        self.max_num_transitions = self.min_num_transitions * 3

    def _fetch_episodes(self):

        rollout_spec = self.algorithm.get_rollout_spec()
        self.replay_buffer = OnpolicyRolloutBuffer(
            state_space=self.env_spec.state_space,
            action_space=self.env_spec.action_space,
            capacity=self.max_num_transitions,
            **rollout_spec
        )

        start_time = time.time()

        while self._num_trajectories < self.min_num_trajectories \
                and self._num_transitions < self.min_num_transitions:

            trajectories_percentrage = \
                100 * self._num_trajectories / self.min_num_trajectories
            trajectories_stats = \
                f"{self._num_trajectories:09d} / " \
                f"{self.min_num_trajectories:09d} " \
                f"({trajectories_percentrage:5.2f}%)"
            transitions_percentrage = \
                100 * self._num_transitions / self.min_num_transitions
            transitions_stats = \
                f"{self._num_transitions:09d} / " \
                f"{self.min_num_transitions:09d} " \
                f"({transitions_percentrage:5.2f}%)"
            print(
                f"trajectories, {trajectories_stats}\t"
                f"transitions, {transitions_stats}\t"
            )

            try:
                episode = self.db_server.get_trajectory()
                assert episode is not None
            except Exception:
                time.sleep(0.5)
                continue

            self._num_trajectories += 1
            self._num_transitions += len(episode[-1])

            observations, actions, rewards, _ = episode
            states = _get_states_from_observations(
                observations, self.env_spec.history_len)
            rollout = self.algorithm.get_rollout(states, actions, rewards)
            self.replay_buffer.push_rollout(
                state=states,
                action=actions,
                reward=rewards,
                **rollout,
            )

        # @TODO: refactor
        self.algorithm.postprocess_buffer(
            self.replay_buffer.buffers,
            len(self.replay_buffer))

        elapsed_time = time.time() - start_time
        self.logger.add_scalar("fetch time", elapsed_time, self.epoch)

    def __run_epoch(self):
        sampler = OnpolicyRolloutSampler(
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
        return metrics

    def _run_train_loop(self):
        while True:
            # start samplers
            self.db_server.set_sample_flag(sample=True)
            # get trajectories
            self._fetch_episodes()
            # stop samplers
            self.db_server.set_sample_flag(sample=False)

            # train & update
            self._run_epoch()

            # cleanup trajectories
            self.db_server.clean_trajectories()
            self._num_trajectories = 0
            self._num_transitions = 0
            del self.replay_buffer
