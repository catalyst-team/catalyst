import os
import gc
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.utils import \
    OnpolicyRolloutBuffer, OnpolicyRolloutSampler, \
    _get_states_from_observations, _make_tuple
from catalyst.rl.db.core import DBSpec
from catalyst.rl.environments.core import EnvironmentSpec
from catalyst.rl.onpolicy.algorithms.core import AlgorithmSpec


class Trainer:
    def __init__(
        self,
        algorithm: AlgorithmSpec,
        env_spec: EnvironmentSpec,
        db_server: DBSpec,
        logdir: str,
        num_workers: int = 1,
        batch_size: int = 64,
        num_mini_epochs: int = 10,
        min_num_trajectories: int = 100,
        min_num_transitions: int = 8192,
        save_period: int = 10,
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
        self.step = 0
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
        self.max_num_transitions = min_num_transitions * 3

        self.save_period = save_period

        self._num_trajectories = 0
        self._num_transitions = 0

        self._sampler_weight_mode = "actor"
        self._gc_period = gc_period

        self.replay_buffer = None

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
        return []

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

    def _update_samplers_weights(self):
        mode = self._sampler_weight_mode
        state_dict = self.algorithm.__dict__[mode].state_dict()
        state_dict = {
            k: v.detach().cpu().numpy()
            for k, v in state_dict.items()
        }
        self.db_server.dump_weights(
            weights=state_dict,
            prefix=mode,
            epoch=self.epoch
        )

    def _train(self):
        start_time = time.time()

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

        for i, batch in enumerate(loader):
            metrics = self.algorithm.train(
                batch,
                actor_update=(self.step % self.actor_grad_period == 0),
                critic_update=(self.step % self.critic_grad_period == 0)
            )

            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.add_scalar(key, value, self.step)

            self.step += 1

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

        self.epoch += 1
        self.save()
        self._update_samplers_weights()

    def _start_train_loop(self):
        while True:
            # start samplers
            self.db_server.set_sample_flag(sample=True)
            # get trajectories
            self._fetch_episodes()

            # stop samplers
            self.db_server.set_sample_flag(sample=False)

            # train & update
            self._train()

            # cleanup trajectories
            self.db_server.clean_trajectories()
            self._num_trajectories = 0
            self._num_transitions = 0
            del self.replay_buffer
            if self.epoch % self._gc_period == 0:
                gc.collect()

    def run(self):
        self._update_samplers_weights()
        self._start_train_loop()
