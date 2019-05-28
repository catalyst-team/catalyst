from typing import Union, List

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc  # noqa E402
import time  # noqa E402
import random  # noqa E402
from datetime import datetime  # noqa E402

import torch  # noqa E402
torch.set_num_threads(1)

from tensorboardX import SummaryWriter  # noqa E402

from catalyst.utils.misc import set_global_seed  # noqa E402
from catalyst.dl.utils import UtilsFactory  # noqa E402
from catalyst.rl.utils import EpisodeRunner  # noqa E402
from catalyst.rl.exploration import ExplorationHandler  # noqa E402
from catalyst.rl.environments.core import EnvironmentSpec  # noqa E402
from catalyst.rl.db.core import DBSpec  # noqa E402
from catalyst.rl.agents.core import ActorSpec, CriticSpec  # noqa E402


_BIG_NUM = int(2 ** 32 - 2)
_SEED_RANGE = _BIG_NUM


class Sampler:
    def __init__(
        self,
        agent: Union[ActorSpec, CriticSpec],
        env: EnvironmentSpec,
        db_server: DBSpec = None,
        exploration_handler: ExplorationHandler = None,
        logdir: str = None,
        id: int = 0,
        mode: str = "infer",
        seeds: List = None,
        max_trajectories_to_sample: int = None,
        force_store: bool = False,
        gc_period: int = 10,
    ):
        self._device = UtilsFactory.prepare_device()
        self._seed = 42 + id
        self._sampler_id = id

        self._infer = mode == "infer"
        self.seeds = seeds

        # logging
        self._prepare_logger(logdir, mode)

        # environment, model, exploration & action handlers
        self.env = env
        self.agent = agent
        self.exploration_handler = exploration_handler
        self.episode_index = 0
        self.episode_runner = EpisodeRunner(
            env=self.env,
            agent=self.agent,
            device=self._device,
            deterministic=self._infer
        )

        # synchronization configuration
        self.db_server = db_server
        self.episode_limit = max_trajectories_to_sample or _BIG_NUM
        self._force_store = force_store
        self._sampler_weight_mode = "actor"
        self._gc_period = gc_period

    def _prepare_logger(self, logdir, mode):
        if logdir is not None:
            current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
            logpath = f"{logdir}/" \
                f"sampler-{mode}-{self._sampler_id}-{current_date}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_checkpoint(
        self,
        *,
        filepath: str = None,
        db_server: DBSpec = None
    ):
        if filepath is not None:
            checkpoint = UtilsFactory.load_checkpoint(filepath)
            weights = checkpoint[f"{self._sampler_weight_mode}_state_dict"]
            self.agent.load_state_dict(weights)
        elif db_server is not None:
            while not db_server.get_sample_flag():
                time.sleep(1.0)
            weights = db_server.load_weights(prefix=self._sampler_weight_mode)
            weights = {k: self._to_tensor(v) for k, v in weights.items()}
            self.agent.load_state_dict(weights)
        else:
            raise NotImplementedError

        self.agent.to(self._device)
        self.agent.eval()

    def _store_trajectory(self):
        if self.db_server is None:
            return
        if not self.db_server.get_sample_flag():
            return

        trajectory = self.episode_runner.get_trajectory()
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

    def _log_to_console(
        self,
        *,
        episode_reward,
        num_steps,
        elapsed_time,
        seed
    ):
        print(
            f"--- episode {int(self.episode_index):05d}:\t"
            f"steps: {int(num_steps):05d}\t"
            f"reward: {episode_reward:9.4f}\t"
            f"time: {elapsed_time:9.4f}\t"
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
            self.load_checkpoint(db_server=self.db_server)
            seed = self._prepare_seed()
            exploration_strategy = \
                self.exploration_handler.get_exploration_strategy() \
                if self.exploration_handler is not None \
                else None
            self.episode_runner.reset(exploration_strategy)

            start_time = time.time()
            episode_info = self.episode_runner.run(
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
            if self.episode_index % self._gc_period == 0:
                gc.collect()
            if self.episode_index >= self.episode_limit:
                return
