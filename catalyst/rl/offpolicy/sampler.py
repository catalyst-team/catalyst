from typing import List

import os
import time
import copy
import random
from datetime import datetime
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from catalyst.utils.misc import set_global_seed
from catalyst.dl.utils import UtilsFactory
from catalyst.rl.offpolicy.utils import EnvWrapper
from catalyst.rl.offpolicy.exploration import ExplorationHandler
from catalyst.rl.environments.core import EnvironmentSpec
from catalyst.rl.db.core import DBSpec

# speed up optimization
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
_BIG_NUM = int(2 ** 32 - 2)
_SEED_RANGE = _BIG_NUM


class Sampler:
    def __init__(
        self,
        network: nn.Module,  # @TODO: actor policy specification?
        env: EnvironmentSpec,
        db_server: DBSpec = None,
        logdir: str = None,
        id: int = 0,
        mode: str = "infer",
        buffer_size: int = int(1e4),
        weights_sync_period: int = 1,
        seeds: List = None,
        episode_limit: int = None,
        force_store: bool = False,
        resume: str = None,
    ):

        self._seed = 42 + id
        set_global_seed(self._seed)
        self._sampler_id = id

        # logging
        self._prepare_logger(logdir, mode)

        # synchronization configuration
        self.db_server = db_server
        self.weights_sync_period = weights_sync_period
        self.episode_limit = episode_limit or _BIG_NUM
        self._force_store = force_store

        # other
        self._infer = mode == "infer"
        self.seeds = seeds

        # environment, model, exploration & action handlers
        self.env = env
        self._device = UtilsFactory.prepare_device()
        self.network = copy.deepcopy(network).to(self._device)
        self.exploration_handler: ExplorationHandler = ExplorationHandler()
        self.episode_index = 0
        self.env_wrapper = EnvWrapper(
            env=self.env,
            actor=self.network,
            capacity=buffer_size,
            deterministic=self._infer
        )

        self._sampler_weight_mode = \
            "critic" if env.discrete_actions else "actor"

        # resume
        if resume is not None:
            self.load_checkpoint(resume=resume)

    def _prepare_logger(self, logdir, mode):
        if logdir is not None:
            current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
            logpath = f"{logdir}/sampler-{mode}-{id}-{current_date}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_checkpoint(self, *, resume: str = None, db_server: DBSpec = None):
        if resume is not None:
            checkpoint = UtilsFactory.load_checkpoint(resume)
            weights = checkpoint[f"{self._sampler_weight_mode}_state_dict"]
            self.network.load_state_dict(weights)
        elif db_server is not None:
            weights = db_server.load_weights(suffix=self._sampler_weight_mode)
            weights = {k: self._to_tensor(v) for k, v in weights.items()}
            self.network.load_state_dict(weights)
        else:
            raise NotImplementedError
        self.network.eval()

    def _store_trajectory(self):
        if self.db_server is None:
            return
        trajectory = self.env_wrapper.get_trajectory(tolist=True)
        self.db_server.push_trajectory(trajectory)

    def _prepare_seed(self):
        seed = self._seed + random.randrange(_SEED_RANGE)
        set_global_seed(seed)
        if self.seeds is None:
            seed = random.randrange(_SEED_RANGE)
        else:
            seed = random.choice(self.seeds)
        set_global_seed(seed)
        return seed

    def _log_to_console(self, *, episode_reward, num_steps, elapsed_time, seed):
        print(
            f"--- episode {self.episode_index:5d}:\t"
            f"steps: {num_steps:5d}\t"
            f"reward: {episode_reward:9.4f}\t"
            f"time: {elapsed_time:5d}\t"
            f"seed: {seed}"
        )

    def _log_to_tensorboard(self, *, episode_reward, num_steps, elapsed_time):
        if self.logger is not None:
            self.logger.add_scalar(
                "episode/num_steps", num_steps, self.episode_index
            )
            self.logger.add_scalar(
                "episode/reward", episode_reward, self.episode_index
            )
            self.logger.add_scalar(
                "time/episode per minute", 60. / elapsed_time,
                self.episode_index
            )
            self.logger.add_scalar(
                "time/steps per second", num_steps / elapsed_time,
                self.episode_index
            )
            self.logger.add_scalar(
                "time/episode time (sec)", elapsed_time, self.episode_index
            )
            self.logger.add_scalar(
                "time/step time (sec)", elapsed_time / num_steps,
                self.episode_index
            )

    def run(self):
        while True:
            if self.episode_index % self.weights_sync_period == 0:
                self.load_checkpoint(db_server=self.db_server)
            seed = self._prepare_seed()
            exploration_strategy = \
                self.exploration_handler.get_exploration_strategy()
            self.env_wrapper.reset(exploration_strategy)

            start_time = time.time()
            episode_info = self.env_wrapper.play_episode(
                exploration_strategy=exploration_strategy)
            elapsed_time = time.time() - start_time

            if not self._infer or self._force_store:
                self._store_trajectory()

            self._log_to_console(
                **episode_info,
                elapsed_time=elapsed_time,
                seed=seed)

            self._log_to_tensorboard(
                **episode_info,
                elapsed_time=elapsed_time)

            self.episode_index += 1
            if self.episode_index >= self.episode_limit:
                return
