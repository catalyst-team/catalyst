import os
import time
import copy
import random
import numpy as np
import torch
from datetime import datetime
from tensorboardX import SummaryWriter

from catalyst.utils.misc import set_global_seed
from catalyst.dl.utils import UtilsFactory
from catalyst.utils.serialization import serialize, deserialize
from catalyst.rl.offpolicy.utils import EnvWrapper, ActionHandler
from catalyst.rl.offpolicy.exploration import ExplorationHandler
from catalyst.rl.offpolicy.exploration.strategies import ParameterSpaceNoise

# speed up optimization
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
_BIG_NUM = int(2 ** 32 - 2)
_SEED_RANGE = _BIG_NUM


class Sampler:
    def __init__(
        self,
        network,
        env,
        id,
        logdir=None,
        redis_server=None,
        redis_prefix=None,
        buffer_size=int(1e4),
        history_len=1,
        critic_distribution=None,
        values_range=(-10., 10.),
        weights_sync_period=1,
        mode="infer",
        resume=None,
        seeds=None,
        action_clip=None,
        episode_limit=None,
        force_store=False,
        discrete_actions=False
    ):

        self._seed = 42 + id
        set_global_seed(self._seed)
        self._sampler_id = id

        # logging
        if logdir is not None:
            current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
            logpath = f"{logdir}/sampler-{mode}-{id}-{current_date}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

        # environment, model, exploration & action handlers
        self.env = env
        self._device = UtilsFactory.prepare_device()
        self.network = copy.deepcopy(network).to(self._device)
        self.exploration_handler: ExplorationHandler = ExplorationHandler()

        n_atoms = 1 if not self.discrete_actions else self.network.out_features
        self.action_handler = ActionHandler(
            device=self._device,
            discrete_actions=self.discrete_actions,
            deterministic=self._infer,
            critic_distribution=critic_distribution,
            n_atoms=n_atoms,
            values_range=values_range,
            action_clip=action_clip
        )

        # main attributes
        self.history_len = history_len
        self.buffer_size = buffer_size
        self.weights_sync_period = weights_sync_period
        self.episode_index = 0

        self.trajectory_buffer: EnvWrapper = None

        # synchronization configuration
        self.redis_server = redis_server
        self.redis_prefix = redis_prefix or ""
        self.episode_limit = episode_limit or _BIG_NUM
        self.force_store = force_store

        self.discrete_actions = discrete_actions
        self._sampler_weight_mode = \
            "critic" if self.discrete_actions else "actor"

        # other
        self._infer = mode == "infer"
        self.seeds = seeds

        # resume
        if resume is not None:
            self.load_checkpoint(resume=resume)

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_checkpoint(self, *, resume=None, redis_server=None):
        if resume is not None:
            checkpoint = UtilsFactory.load_checkpoint(resume)
            weights = checkpoint[f"{self._sampler_weight_mode}_state_dict"]
            self.network.load_state_dict(weights)
        elif redis_server is not None:
            weights = deserialize(
                redis_server.get(
                    f"{self.redis_prefix}_{self._sampler_weight_mode}_weights"
                )
            )
            weights = {k: self._to_tensor(v) for k, v in weights.items()}
            self.network.load_state_dict(weights)
        else:
            raise NotImplementedError
        self.network.eval()

    def _store_trajectory(self):
        if self.redis_server is None:
            return
        trajectory = self.env_wrapper.get_trajectory(tolist=True)
        trajectory = serialize(trajectory)
        self.redis_server.rpush("trajectories", trajectory)

    def _prepare_seed(self):
        seed = self._seed + random.randrange(_SEED_RANGE)
        set_global_seed(seed)
        if self.seeds is None:
            seed = random.randrange(_SEED_RANGE)
        else:
            seed = random.choice(self.seeds)
        set_global_seed(seed)
        return seed

    def _prepare_exploration_strategy(self):
        exploration_strategy = \
            self.exploration_handler.get_exploration_strategy()
        if isinstance(exploration_strategy, ParameterSpaceNoise) \
                and self.episode_index > 1:
            states = self.trajectory_buffer._get_states_history(
                history_len=self.history_len
            )
            states = self._to_tensor(states).detach()
            exploration_strategy.update_actor(self.network, states)
        return exploration_strategy

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
        self.episode_index = 0

        while True:
            if self.episode_index % self.weights_sync_period == 0:
                self.load_checkpoint(redis_server=self.redis_server)

            seed = self._prepare_seed()
            exploration_strategy = self._prepare_exploration_strategy()
            self.env_wrapper = EnvWrapper(
                env=self.env,
                capacity=self.buffer_size
            )

            start_time = time.time()
            episode_info = self.env_wrapper.play_episode(
                actor=self.network,
                exploration_strategy=exploration_strategy)
            elapsed_time = time.time() - start_time

            if not self._infer or self.force_store:
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
