from typing import Union, List, Dict

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
        **kwargs
    ):
        self._device = utils.get_device()
        self._sampler_id = id

        self._is_infer = mode in ["valid", "infer"]
        self.seeds = seeds
        self._seeder = Seeder(init_seed=42 + id)

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
            deterministic=self._is_infer,
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
        self.checkpoint = None

        #  special
        self._init(**kwargs)

    def _init(self, **kwargs):
        assert len(kwargs) == 0

    def _prepare_logger(self, logdir, mode):
        if logdir is not None:
            timestamp = datetime.utcnow().strftime("%y%m%d.%H%M%S")
            logpath = f"{logdir}/" \
                f"sampler.{mode}.{self._sampler_id}.{timestamp}"
            os.makedirs(logpath, exist_ok=True)
            self.logdir = logpath
            self.logger = SummaryWriter(logpath)
        else:
            self.logdir = None
            self.logger = None

    def _start_db_loop(self):
        self._db_loop_thread = threading.Thread(
            target=_db2sampler_loop,
            kwargs={
                "sampler": self,
            }
        )
        self._db_loop_thread.start()

    def load_checkpoint(
        self,
        *,
        filepath: str = None,
        db_server: DBSpec = None
    ):
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
        elif db_server is not None:
            checkpoint = db_server.load_checkpoint()
            while checkpoint is None:
                time.sleep(3.0)
                checkpoint = db_server.load_checkpoint()
        else:
            raise NotImplementedError

        self.checkpoint = checkpoint
        weights = self.checkpoint[f"{self._weights_sync_mode}_state_dict"]
        weights = {
            k: utils.any2device(v, device=self._device)
            for k, v in weights.items()}
        self.agent.load_state_dict(weights)
        self.agent.to(self._device)
        self.agent.eval()

    def _store_trajectory(self, trajectory, raw=False):
        if self.db_server is None:
            return
        self.db_server.push_trajectory(trajectory, raw=raw)

    def _get_seed(self):
        if self.seeds is not None:
            seed = self.seeds[self.trajectory_index % len(self.seeds)]
        else:
            seed = self._seeder()[0]
        set_global_seed(seed)
        return seed

    def _log_to_console(
        self,
        *,
        reward,
        raw_reward,
        num_steps,
        elapsed_time,
        seed
    ):
        metrics = [
            f"trajectory {int(self.trajectory_index):05d}",
            f"steps: {int(num_steps):05d}",
            f"reward: {reward:9.3f}",
            f"raw_reward: {raw_reward:9.3f}",
            f"time: {elapsed_time:9.3f}",
            f"seed: {seed:010d}",
        ]
        metrics = " | ".join(metrics)
        print(f"--- {metrics}")

    def _log_to_tensorboard(
        self,
        *,
        reward,
        raw_reward,
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
                "trajectory/raw_reward", raw_reward, self.trajectory_index
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
            raw_trajectory = trajectory_info.pop("raw_trajectory", None)
            # Do it firsthand, so the loggers don't crush
            if not self._is_infer or self._force_store:
                self._store_trajectory(trajectory)
                if raw_trajectory is not None:
                    self._store_trajectory(raw_trajectory, raw=True)
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


class ValidSampler(Sampler):

    def _init(self, save_n_best: int = 3, **kwargs):
        assert len(kwargs) == 0
        self.save_n_best = save_n_best
        self.best_agents = []
        self._sample_flag.value = True

    def load_checkpoint(
        self,
        *,
        filepath: str = None,
        db_server: DBSpec = None
    ):
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
        elif db_server is not None:
            current_epoch = db_server.epoch
            checkpoint = db_server.load_checkpoint()
            while checkpoint is None or db_server.epoch <= current_epoch:
                time.sleep(3.0)
                checkpoint = db_server.load_checkpoint()
        else:
            raise NotImplementedError

        self.checkpoint = checkpoint
        weights = self.checkpoint[f"{self._weights_sync_mode}_state_dict"]
        weights = {
            k: utils.any2device(v, device=self._device)
            for k, v in weights.items()}
        self.agent.load_state_dict(weights)
        self.agent.to(self._device)
        self.agent.eval()

    @staticmethod
    def rewards2metric(rewards):
        return np.mean(rewards)  # - np.std(rewards)

    def save_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        save_n_best: int = 5,
        minimize_metric: bool = False
    ):
        agent_rewards = checkpoint["rewards"]
        agent_metric = self.rewards2metric(agent_rewards)

        is_best = len(self.best_agents) == 0 or \
            agent_metric > self.rewards2metric(self.best_agents[0][1])
        suffix = f"{checkpoint['epoch']}"
        filepath = utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        self.best_agents.append((filepath, agent_rewards))
        self.best_agents = sorted(
            self.best_agents,
            key=lambda x: x[1],
            reverse=not minimize_metric
        )
        if len(self.best_agents) > save_n_best:
            last_item = self.best_agents.pop(-1)
            last_filepath = last_item[0]
            os.remove(last_filepath)

    def _run_sample_loop(self):
        assert self.seeds is not None

        while True:
            self.load_checkpoint(db_server=self.db_server)
            trajectories_rewards = []

            for i in range(len(self.seeds)):
                trajectory, trajectory_info = self._run_trajectory_loop()
                trajectories_rewards.append(trajectory_info["reward"])

                self._log_to_console(**trajectory_info)
                self._log_to_tensorboard(**trajectory_info)
                self.trajectory_index += 1

                if self.trajectory_index % self._gc_period == 0:
                    gc.collect()

            self.checkpoint["rewards"] = trajectories_rewards
            self.checkpoint["epoch"] = self.db_server.epoch
            self.save_checkpoint(
                logdir=self.logdir,
                checkpoint=self.checkpoint,
                save_n_best=self.save_n_best
            )

    def run(self):
        self._start_sample_loop()


def _db2sampler_loop(sampler: Sampler):
    while True:
        sampler._sample_flag.value = sampler.db_server.get_sample_flag()
        time.sleep(5.0)


__all__ = ["Sampler", "ValidSampler"]
