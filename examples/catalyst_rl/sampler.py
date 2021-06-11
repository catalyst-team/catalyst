# flake8: noqa
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import gc
import time

import gym
from torch import nn

from catalyst import utils

if TYPE_CHECKING:
    from db import IRLDatabase
    from misc import Trajectory


class ISampler(ABC):
    def __init__(
        self,
        env: gym.Env,
        actor: nn.Module,
        db_server: "IRLDatabase",
        sampler_index: int,
        weights_key: str,
        weights_sync_period: int,
        device=None,
    ):
        self.env = env
        self.actor = actor
        self.db_server = db_server
        self._weights_key = weights_key
        self._weights_sync_period = weights_sync_period
        self.sampler_index = sampler_index
        self.device = device or "cpu"
        self.trajectory_index = 0

    @abstractmethod
    def get_trajectory(
        self,
        env: gym.Env,
        actor: nn.Module,
        device,
        sampler_index: int = None,
        trajectory_index: int = None,
    ) -> "Trajectory":
        pass

    def _get_trajectory(self) -> "Trajectory":
        return self.get_trajectory(
            sampler_index=self.sampler_index,
            trajectory_index=self.trajectory_index,
            env=self.env,
            actor=self.actor,
            device=self.device,
        )

    def _load_checkpoint(self, *, filepath: str = None, db_server: "IRLDatabase" = None):
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
        elif db_server is not None:
            checkpoint = db_server.get_checkpoint()
            while checkpoint is None:
                time.sleep(3.0)
                checkpoint = db_server.get_checkpoint()
        else:
            raise NotImplementedError("No checkpoint found")

        # self.checkpoint = checkpoint
        # weights = utils.any2device(checkpoint[self._weights_key], device=self.device)
        # weights = {k: utils.any2device(v, device=self.device) for k, v in weights.items()}
        weights = checkpoint[self._weights_key]
        self.actor.load_state_dict(weights)
        self.actor.to(self.device)
        self.actor.eval()

    def _store_trajectory(self, trajectory):
        self.db_server.add_trajectory(trajectory)

    def _run(self):
        while True:
            # 1 – load from db, 2 – resume load trick (already have checkpoint)
            # need_checkpoint = self.db_server is not None or self.checkpoint is None
            if self.trajectory_index % self._weights_sync_period == 0:
                self._load_checkpoint(db_server=self.db_server)

            trajectory = self._get_trajectory()
            self._store_trajectory(trajectory)
            self.trajectory_index += 1

            if self.trajectory_index % self._weights_sync_period == 0:
                gc.collect()

    def run(self):
        self._run()
