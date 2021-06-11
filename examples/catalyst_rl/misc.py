# flake8: noqa
from typing import Callable, Dict, TYPE_CHECKING
from collections import namedtuple
import copy
import multiprocessing as mp
import threading
import time

import numpy as np
import torch.nn as nn

from catalyst import dl

if TYPE_CHECKING:
    from buffer import OffpolicyReplayBuffer
    from db import IRLDatabase

Trajectory = namedtuple("Trajectory", field_names=["observations", "actions", "rewards", "dones"])


def structed2dict(array: np.ndarray):
    if isinstance(array, (np.ndarray, np.void)) and array.dtype.fields is not None:
        array = {key: array[key] for key in array.dtype.fields.keys()}
    return array


def dict2structed(array: Dict):
    if isinstance(array, dict):
        capacity = 0
        dtype = []
        for key, value in array.items():
            capacity = len(value)
            dtype.append((key, value.dtype, value.shape[1:]))
        dtype = np.dtype(dtype)

        array_ = np.empty(capacity, dtype=dtype)
        for key, value in array.items():
            array_[key] = value
        array = array_

    return array


def structed2dict_trajectory(trajectory: Trajectory):
    observations, actions, rewards, dones = (
        structed2dict(trajectory.observations),
        structed2dict(trajectory.actions),
        trajectory.rewards,
        trajectory.dones,
    )
    trajectory = Trajectory(observations, actions, rewards, dones)
    return trajectory


def dict2structed_trajectory(trajectory: Trajectory):
    observations, actions, rewards, dones = (
        dict2structed(trajectory.observations),
        dict2structed(trajectory.actions),
        trajectory.rewards,
        trajectory.dones,
    )
    trajectory = Trajectory(observations, actions, rewards, dones)
    return trajectory


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Updates the `target` data with the `source` one smoothing by ``tau`` (inplace operation)."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def db2buffer_loop(
    db_server: "IRLDatabase", buffer: "OffpolicyReplayBuffer",
):
    trajectory = None
    while True:
        try:
            if trajectory is None:
                trajectory: Trajectory = db_server.get_trajectory()

            if trajectory is not None:
                if buffer.add_trajectory(trajectory):
                    trajectory = None
                else:
                    time.sleep(1.0)
            else:
                if not db_server.training_enabled:
                    print("THE END.")
                    return
                time.sleep(1.0)
        except Exception as ex:
            print("=" * 80)
            print("Something go wrong with trajectory:")
            print(ex)
            print(trajectory)
            print("=" * 80)
            trajectory = None


def run_sampler(sampler_fn, *args, **kwargs):
    sampler = sampler_fn(*args, **kwargs)
    sampler.run()


class GameCallback(dl.Callback):
    def __init__(
        self,
        *,
        sampler_fn: Callable,
        env,
        replay_buffer: "OffpolicyReplayBuffer",
        db_server: "IRLDatabase",
        actor_key: str,
        num_samplers: int = 1,
        min_transactions_num: int = int(1e3),
    ):
        super().__init__(order=0)
        self.sampler_fn = sampler_fn
        self.env = env
        self.replay_buffer = replay_buffer
        self.db_server = db_server
        self.actor_key = actor_key
        self.num_samplers = num_samplers
        self.min_transactions_num = min_transactions_num

        self.samplers = []

        self._db_loop_thread = threading.Thread(
            target=db2buffer_loop,
            kwargs={"db_server": self.db_server, "buffer": self.replay_buffer},
        )

    def _sync_checkpoint(self, runner: dl.IRunner):
        actor = copy.deepcopy(runner.model[self.actor_key]).to("cpu")
        checkpoint = {self.actor_key: actor.state_dict()}
        self.db_server.add_checkpoint(checkpoint=checkpoint, epoch=runner.stage_epoch_step)

    def _fetch_initial_buffer(self):
        buffer_size = self.replay_buffer.length
        while buffer_size < self.min_transactions_num:
            self.replay_buffer.recalculate_index()

            num_trajectories = self.replay_buffer.num_trajectories
            num_transitions = self.replay_buffer.num_transitions
            buffer_size = self.replay_buffer.length

            metrics = [
                f"fps: {0:7.1f}",
                f"updates per sample: {0:7.1f}",
                f"trajectories: {num_trajectories:09d}",
                f"transitions: {num_transitions:09d}",
                f"buffer size: " f"{buffer_size:09d}/{self.min_transactions_num:09d}",
            ]
            metrics = " | ".join(metrics)
            print(f"--- {metrics}")

            time.sleep(1.0)

    def on_stage_start(self, runner: dl.IRunner) -> None:
        # db sync
        self._sync_checkpoint(runner=runner)
        # self.db_server.add_message(IRLDatabaseMessage.ENABLE_TRAINING)  # deprecated?
        # self.db_server.add_message(IRLDatabaseMessage.ENABLE_SAMPLING)  # deprecated?

        # samplers
        for i in range(self.num_samplers):
            p = mp.Process(
                target=run_sampler,
                kwargs=dict(
                    sampler_fn=self.sampler_fn,
                    env=copy.deepcopy(self.env),
                    actor=copy.deepcopy(runner.model[self.actor_key]).to("cpu"),
                    db_server=self.db_server,
                    sampler_index=i,
                    weights_key=self.actor_key,
                    weights_sync_period=10,
                    device="cpu",
                ),
                daemon=True,
            )
            p.start()
            self.samplers.append(p)

        # for p in self.samplers:
        #     p.join()

        # db -> local storage
        self._db_loop_thread.start()
        # init local storage
        self._fetch_initial_buffer()

    def on_epoch_end(self, runner: dl.IRunner):
        runner.epoch_metrics["_epoch_"]["num_trajectories"] = self.replay_buffer.num_trajectories
        runner.epoch_metrics["_epoch_"]["num_transitions"] = self.replay_buffer.num_transitions
        runner.epoch_metrics["_epoch_"]["updates_per_sample"] = (
            runner.loader_sample_step / self.replay_buffer.num_transitions
        )
        runner.epoch_metrics["_epoch_"]["reward"] = np.mean(
            self.replay_buffer._trajectories_rewards[-100:]
        )
        self._sync_checkpoint(runner=runner)
        self.replay_buffer.recalculate_index()

    def on_stage_end(self, runner: dl.IRunner) -> None:
        from db import IRLDatabaseMessage

        for p in self.samplers:
            p.terminate()
        self.db_server.add_message(IRLDatabaseMessage.DISABLE_TRAINING)
        self.db_server.add_message(IRLDatabaseMessage.DISABLE_SAMPLING)
