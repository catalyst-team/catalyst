from typing import Union, List, Dict

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc  # noqa E402
import time  # noqa E402
import shutil  # noqa E402
from pathlib import Path  # noqa E402
import threading  # noqa E402
from ctypes import c_bool  # noqa E402
import multiprocessing as mp  # noqa E402
import numpy as np  # noqa E402
import logging  # noqa E402

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

logger = logging.getLogger(__name__)

if os.environ.get("USE_WANDB", "1") == "1":
    try:
        import wandb
        WANDB_ENABLED = True
    except ImportError:
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
        WANDB_ENABLED = False
else:
    WANDB_ENABLED = False


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
        deterministic: bool = None,
        weights_sync_period: int = 1,
        weights_sync_mode: str = None,
        seeds: List = None,
        trajectory_limit: int = None,
        force_store: bool = False,
        gc_period: int = 10,
        monitoring_params: Dict = None,
        **kwargs
    ):
        self._device = utils.get_device()
        self._sampler_id = id

        self._deterministic = deterministic \
            if deterministic is not None \
            else mode in ["valid", "infer"]
        self.seeds = seeds
        self._seeder = Seeder(init_seed=42 + id)

        # logging
        self._prepare_logger(logdir, mode)
        self._sampling_flag = mp.Value(c_bool, False)
        self._training_flag = mp.Value(c_bool, True)

        # environment, model, exploration & action handlers
        self.env = env
        self.agent = agent
        self.exploration_handler = exploration_handler
        self.trajectory_index = 0
        self.trajectory_sampler = TrajectorySampler(
            env=self.env,
            agent=self.agent,
            device=self._device,
            deterministic=self._deterministic,
            sampling_flag=self._sampling_flag
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
        self.monitoring_params = monitoring_params
        self._init(**kwargs)

    def _init(self, **kwargs):
        global WANDB_ENABLED
        assert len(kwargs) == 0
        if WANDB_ENABLED:
            if self.monitoring_params is not None:
                self.checkpoints_glob: List[str] = \
                    self.monitoring_params.pop(
                        "checkpoints_glob", ["best.pth", "last.pth"])

                wandb.init(**self.monitoring_params)
            else:
                WANDB_ENABLED = False
        self.wandb_mode = "sampler"

    def _prepare_logger(self, logdir, mode):
        if logdir is not None:
            timestamp = utils.get_utcnow_time()
            logpath = f"{logdir}/" \
                f"sampler.{mode}.{self._sampler_id}.{timestamp}"
            os.makedirs(logpath, exist_ok=True)
            self.logdir = logpath
            self.logger = SummaryWriter(logpath)
        else:
            self.logdir = None
            self.logger = None

    def _start_db_loop(self):
        if self.db_server is None:
            self._training_flag.value = True
            self._sampling_flag.value = True
            return
        self._db_loop_thread = threading.Thread(
            target=_db2sampler_loop, kwargs={
                "sampler": self,
            }
        )
        self._db_loop_thread.start()

    def load_checkpoint(
        self, *, filepath: str = None, db_server: DBSpec = None
    ):
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
        elif db_server is not None:
            checkpoint = db_server.get_checkpoint()
            while checkpoint is None:
                time.sleep(3.0)
                checkpoint = db_server.get_checkpoint()
        else:
            raise NotImplementedError("No checkpoint found")

        self.checkpoint = checkpoint
        weights = self.checkpoint[f"{self._weights_sync_mode}_state_dict"]
        weights = {
            k: utils.any2device(v, device=self._device)
            for k, v in weights.items()
        }
        self.agent.load_state_dict(weights)
        self.agent.to(self._device)
        self.agent.eval()

    def _store_trajectory(self, trajectory, raw=False):
        if self.db_server is None:
            return
        self.db_server.put_trajectory(trajectory, raw=raw)

    def _get_seed(self):
        if self.seeds is not None:
            seed = self.seeds[self.trajectory_index % len(self.seeds)]
        else:
            seed = self._seeder()[0]
        set_global_seed(seed)
        return seed

    def _log_to_console(
        self, *, reward, raw_reward, num_steps, elapsed_time, seed
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
        self, *, reward, raw_reward, num_steps, elapsed_time, **kwargs
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
                "time/trajectory_time_sec", elapsed_time, self.trajectory_index
            )
            self.logger.add_scalar(
                "time/step_time_sec", elapsed_time / num_steps,
                self.trajectory_index
            )

            self.logger.flush()

    @staticmethod
    def _log_wandb_metrics(
        metrics: Dict,
        step: int,
        mode: str,
        suffix: str = ""
    ):
        metrics = {
            f"{mode}/{key}{suffix}": value
            for key, value in metrics.items()
        }
        step = None  # @TODO: fix, wandb issue
        wandb.log(metrics, step=step)

    def _log_to_wandb(self, *, step, suffix="", **metrics):
        if WANDB_ENABLED:
            self._log_wandb_metrics(
                metrics, step=step, mode=self.wandb_mode, suffix=suffix)

    def _save_wandb(self):
        if WANDB_ENABLED:
            logdir_src = Path(self.logdir)
            logdir_dst = Path(wandb.run.dir)

            events_src = list(logdir_src.glob("events.out.tfevents*"))
            if len(events_src) > 0:
                events_src = events_src[0]
                os.makedirs(f"{logdir_dst}/{logdir_src.name}", exist_ok=True)
                shutil.copy2(
                    f"{str(events_src.absolute())}",
                    f"{logdir_dst}/{logdir_src.name}/{events_src.name}")

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
            exploration_strategy=exploration_strategy
        )
        elapsed_time = time.time() - start_time

        trajectory_info = trajectory_info or {}
        trajectory_info.update({"elapsed_time": elapsed_time, "seed": seed})
        return trajectory, trajectory_info

    def _run_sample_loop(self):
        while self._training_flag.value:
            while not self._sampling_flag.value:
                if not self._training_flag.value:
                    return
                time.sleep(5.0)

            # 1 – load from db, 2 – resume load trick (already have checkpoint)
            need_checkpoint = \
                self.db_server is not None or self.checkpoint is None
            if self.trajectory_index % self._weights_sync_period == 0 \
                    and need_checkpoint:
                self.load_checkpoint(db_server=self.db_server)
                self._save_wandb()

            trajectory, trajectory_info = self._run_trajectory_loop()
            if trajectory is None:
                continue
            raw_trajectory = trajectory_info.pop("raw_trajectory", None)
            # Do it firsthand, so the loggers don't crush
            if not self._deterministic or self._force_store:
                self._store_trajectory(trajectory)
                if raw_trajectory is not None:
                    self._store_trajectory(raw_trajectory, raw=True)
            self._log_to_console(**trajectory_info)
            self._log_to_tensorboard(**trajectory_info)
            self._log_to_wandb(step=self.trajectory_index, **trajectory_info)
            self.trajectory_index += 1

            if self.trajectory_index % self._gc_period == 0:
                gc.collect()

            if not self._training_flag.value \
                    or self.trajectory_index >= self._trajectory_limit:
                return

    def _start_sample_loop(self):
        self._run_sample_loop()

    def run(self):
        self._start_db_loop()
        self._start_sample_loop()

        if self.logger is not None:
            self.logger.close()


class ValidSampler(Sampler):
    @staticmethod
    def rewards2metric(rewards):
        return np.mean(rewards)  # - np.std(rewards)

    def _init(
        self,
        save_n_best: int = 3,
        main_metric: str = "raw_reward",
        main_metric_fn: str = "mean",
        **kwargs
    ):
        assert len(kwargs) == 0
        assert main_metric in ["reward", "raw_reward"]
        assert main_metric_fn in ["mean", "mean-std"]
        super()._init()
        self.wandb_mode = "valid_sampler"

        self.save_n_best = save_n_best
        self.main_metric = main_metric
        self.best_agents = []
        self._sampling_flag.value = True

        if main_metric_fn == "mean":
            self.rewards2metric = lambda x: np.mean(x)
        elif main_metric_fn == "mean-std":
            self.rewards2metric = lambda x: np.mean(x) - np.std(x)
        else:
            raise NotImplementedError()

    def load_checkpoint(
        self, *, filepath: str = None, db_server: DBSpec = None
    ) -> bool:
        if filepath is not None:
            checkpoint = utils.load_checkpoint(filepath)
        elif db_server is not None:
            current_epoch = db_server.epoch
            checkpoint = db_server.get_checkpoint()
            if not db_server.training_enabled \
                    and db_server.epoch == current_epoch:
                return False

            while checkpoint is None or db_server.epoch <= current_epoch:
                time.sleep(3.0)
                checkpoint = db_server.get_checkpoint()

                if not db_server.training_enabled \
                        and db_server.epoch == current_epoch:
                    return False
        else:
            return False

        self.checkpoint = checkpoint
        weights = self.checkpoint[f"{self._weights_sync_mode}_state_dict"]
        weights = {
            k: utils.any2device(v, device=self._device)
            for k, v in weights.items()
        }
        self.agent.load_state_dict(weights)
        self.agent.to(self._device)
        self.agent.eval()

        return True

    def _save_wandb(self):
        super()._save_wandb()
        if WANDB_ENABLED:
            logdir_src = Path(self.logdir)
            logdir_dst = Path(wandb.run.dir)
            checkpoints_src = logdir_src.joinpath("checkpoints")
            checkpoints_dst = logdir_dst.joinpath("checkpoints")
            os.makedirs(checkpoints_dst, exist_ok=True)

            checkpoint_paths = []
            for glob in self.checkpoints_glob:
                checkpoint_paths.extend(list(checkpoints_src.glob(glob)))
            checkpoint_paths = list(set(checkpoint_paths))
            for checkpoint_path in checkpoint_paths:
                shutil.copy2(
                    f"{str(checkpoint_path.absolute())}",
                    f"{checkpoints_dst}/{checkpoint_path.name}")

    def save_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        save_n_best: int = 3,
        main_metric: str = "raw_reward",
        minimize_metric: bool = False
    ):
        agent_rewards = checkpoint[main_metric]
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
            self.best_agents, key=lambda x: x[1], reverse=not minimize_metric
        )
        if len(self.best_agents) > save_n_best:
            last_item = self.best_agents.pop(-1)
            last_filepath = last_item[0]
            os.remove(last_filepath)

    def _run_sample_loop(self):
        assert self.seeds is not None

        while True:
            # 1 – load from db, 2 – resume load trick (already have checkpoint)
            need_checkpoint = \
                self.db_server is not None or self.checkpoint is None
            ok = self.load_checkpoint(db_server=self.db_server) \
                if need_checkpoint \
                else True
            if not ok:
                return

            trajectories_reward, trajectories_raw_reward = [], []

            for i in range(len(self.seeds)):
                trajectory, trajectory_info = self._run_trajectory_loop()
                trajectories_reward.append(trajectory_info["reward"])
                trajectories_raw_reward.append(
                    trajectory_info.get(
                        "raw_reward",
                        trajectory_info["reward"]
                    )
                )
                trajectory_info.pop("raw_trajectory", None)
                self._log_to_console(**trajectory_info)
                self._log_to_tensorboard(**trajectory_info)
                self._log_to_wandb(
                    step=self.trajectory_index, **trajectory_info)
                self.trajectory_index += 1

                if self.trajectory_index % self._gc_period == 0:
                    gc.collect()

            loop_metrics = {
                "trajectory/_mean_valid_reward":
                    self.rewards2metric(trajectories_reward),
                "trajectory/_mean_valid_raw_reward":
                    self.rewards2metric(trajectories_raw_reward),
            }

            if self.logger is not None:
                for key, value in loop_metrics.items():
                    self.logger.add_scalar(key, value, self.db_server.epoch)
            self._log_to_wandb(step=self.db_server.epoch, **loop_metrics)

            self.checkpoint["reward"] = trajectories_reward
            self.checkpoint["raw_reward"] = trajectories_raw_reward
            self.checkpoint["epoch"] = self.db_server.epoch
            self.save_checkpoint(
                logdir=self.logdir,
                checkpoint=self.checkpoint,
                save_n_best=self.save_n_best,
                main_metric=self.main_metric,
            )
            self._save_wandb()

    def run(self):
        self._start_sample_loop()

        if self.logger is not None:
            self.logger.close()


def _db2sampler_loop(sampler: Sampler):
    while True:
        flag = sampler.db_server.sampling_enabled
        sampler._sampling_flag.value = flag
        if not flag and not sampler.db_server.training_enabled:
            sampler._training_flag.value = False
            return
        time.sleep(5.0)


__all__ = ["Sampler", "ValidSampler"]
