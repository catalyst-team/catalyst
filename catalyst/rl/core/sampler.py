from typing import Union, List

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc  # noqa E402
import time  # noqa E402
import threading  # noqa E402
from ctypes import c_bool  # noqa E402
import multiprocessing as mp  # noqa E402
from datetime import datetime  # noqa E402
import numpy as np  # noqa E402

import torch  # noqa E402
torch.set_num_threads(1)

from tensorboardX import SummaryWriter  # noqa E402

from catalyst.utils.seed import set_global_seed, Seeder  # noqa E402
from catalyst import utils  # noqa E402
from .trajectory_sampler import TrajectorySampler  # noqa E402
from .exploration import ExplorationHandler  # noqa E402
from .environment import EnvironmentSpec  # noqa E402
from .db import DBSpec  # noqa E402
from .agent import ActorSpec, CriticSpec  # noqa E402


class Sampler:
    def __init__(
        self,
        agent: Union[ActorSpec, CriticSpec],
        env: EnvironmentSpec,
        db_server: DBSpec = None,
        exploration_handler: ExplorationHandler = None,
        logdir: str = None,
        id: int = 0,
        mode: str = "infer",  # train/valid/infer
        weights_sync_period: int = 1,
        weights_sync_mode: str = None,
        seeds: List = None,
        trajectory_limit: int = None,
        force_store: bool = False,
        gc_period: int = 10,
    ):
        self._device = utils.get_device()
        self._sampler_id = id

        self._infer = mode == "infer"
        self.seeds = seeds
        self._seeder = Seeder(
            init_seed=42 + id,
            max_seed=len(seeds) if seeds is not None else None
        )

        # logging
        self._prepare_logger(logdir, mode)
        self._sample_flag = mp.Value(c_bool, False)

        # environment, model, exploration & action handlers
        self.env = env
        self.agent = agent
        self.exploration_handler = exploration_handler
        self.trajectory_index = 0
        self.trajectory_sampler = TrajectorySampler(
            env=self.env,
            agent=self.agent,
            device=self._device,
            deterministic=self._infer,
            sample_flag=self._sample_flag
        )

        # synchronization configuration
        self.db_server = db_server
        self._weights_sync_period = weights_sync_period
        self._weights_sync_mode = weights_sync_mode
        self._trajectory_limit = trajectory_limit or np.iinfo(np.int32).max
        self._force_store = force_store
        self._gc_period = gc_period
        self._db_loop_thread = None

    def _prepare_logger(self, logdir, mode):
        if logdir is not None:
            timestamp = datetime.utcnow().strftime("%y%m%d.%H%M%S")
            logpath = f"{logdir}/" \
                f"sampler.{mode}.{self._sampler_id}.{timestamp}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

    def _start_db_loop(self):
        self._db_loop_thread = threading.Thread(
            target=_db2sampler_loop,
            kwargs={
                "sampler": self,
            }
        )
        self._db_loop_thread.start()

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_checkpoint(
        self,
        *,
        filepath: str = None,
        db_server: DBSpec = None
    ):
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
            weights = checkpoint[f"{self._weights_sync_mode}_state_dict"]
            self.agent.load_state_dict(weights)
        elif db_server is not None:
            weights = db_server.load_weights(prefix=self._weights_sync_mode)
            while weights is None:
                time.sleep(1.0)
                weights = db_server.load_weights(
                    prefix=self._weights_sync_mode)
            weights = {k: self._to_tensor(v) for k, v in weights.items()}
            self.agent.load_state_dict(weights)
        else:
            raise NotImplementedError

        self.agent.to(self._device)
        self.agent.eval()

    def _store_trajectory(self, trajectory):
        if self.db_server is None:
            return
        self.db_server.push_trajectory(trajectory)

    def _get_seed(self):
        seed = self._seeder()[0]
        if self.seeds is not None:
            seed = self.seeds[seed]
        set_global_seed(seed)
        return seed

    def _log_to_console(
        self,
        *,
        reward,
        num_steps,
        elapsed_time,
        seed
    ):
        print(
            f"--- trajectory {int(self.trajectory_index):05d}:\t"
            f"steps: {int(num_steps):05d}\t"
            f"reward: {reward:9.4f}\t"
            f"time: {elapsed_time:9.4f}\t"
            f"seed: {seed}"
        )

    def _log_to_tensorboard(
        self,
        *,
        reward,
        num_steps,
        elapsed_time,
        **kwargs
    ):
        if self.logger is not None:
            self.logger.add_scalar(
                "trajectory/num_steps", num_steps, self.trajectory_index
            )
            self.logger.add_scalar(
                "trajectory/reward", reward, self.trajectory_index
            )
            self.logger.add_scalar(
                "time/trajectories_per_minute", 60. / elapsed_time,
                self.trajectory_index
            )
            self.logger.add_scalar(
                "time/steps_per_second", num_steps / elapsed_time,
                self.trajectory_index
            )
            self.logger.add_scalar(
                "time/trajectory_time_sec",
                elapsed_time,
                self.trajectory_index
            )
            self.logger.add_scalar(
                "time/step_time_sec", elapsed_time / num_steps,
                self.trajectory_index
            )

    @torch.no_grad()
    def _run_trajectory_loop(self):
        seed = self._get_seed()
        exploration_strategy = \
            self.exploration_handler.get_exploration_strategy() \
            if self.exploration_handler is not None \
            else None
        self.trajectory_sampler.reset(exploration_strategy)

        start_time = time.time()
        trajectory, trajectory_info = self.trajectory_sampler.sample(
            exploration_strategy=exploration_strategy)
        elapsed_time = time.time() - start_time

        trajectory_info = trajectory_info or {}
        trajectory_info.update({"elapsed_time": elapsed_time, "seed": seed})
        return trajectory, trajectory_info

    def _run_sample_loop(self):
        while True:
            while not self._sample_flag.value:
                time.sleep(5.0)

            if self.trajectory_index % self._weights_sync_period == 0:
                self.load_checkpoint(db_server=self.db_server)

            trajectory, trajectory_info = self._run_trajectory_loop()
            if trajectory is None:
                continue

            if not self._infer or self._force_store:
                self._store_trajectory(trajectory)
            self._log_to_console(**trajectory_info)
            self._log_to_tensorboard(**trajectory_info)
            self.trajectory_index += 1

            if self.trajectory_index % self._gc_period == 0:
                gc.collect()

            if self.trajectory_index >= self._trajectory_limit:
                return

    def _start_sample_loop(self):
        self._run_sample_loop()

    def run(self):
        self._start_db_loop()
        self._start_sample_loop()


def _db2sampler_loop(sampler: Sampler):
    while True:
        sampler._sample_flag.value = sampler.db_server.get_sample_flag()
        time.sleep(5.0)


__all__ = ["Sampler"]
