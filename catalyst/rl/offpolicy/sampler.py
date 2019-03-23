import os
import time
import copy
import random
import numpy as np
import torch
from datetime import datetime
from tensorboardX import SummaryWriter

from catalyst.utils.misc import set_global_seeds
from catalyst.dl.utils import UtilsFactory
from catalyst.utils.serialization import serialize, deserialize
from catalyst.rl.random_process import RandomProcess
from catalyst.rl.offpolicy.utils import SamplerBuffer, ActionHandler
from catalyst.rl.offpolicy.exploration import ParameterSpaceNoise

# speed up optimization
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
SEED_RANGE = 2**32 - 2


class Sampler:
    def __init__(
        self,
        network,
        explorator,
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
        action_clip=(-1, 1),
        episode_limit=None,
        force_store=False,
        discrete_actions=False
    ):

        self._seed = 42 + id
        set_global_seeds(self._seed)

        self._sampler_id = id
        self._device = UtilsFactory.prepare_device()
        self.network = copy.deepcopy(network).to(self._device)
        self.explorator = explorator
        self.env = env
        self.redis_server = redis_server
        self.redis_prefix = redis_prefix or ""
        self.resume = resume
        self.episode_limit = episode_limit or int(2**32 - 2)
        self.force_store = force_store
        self.discrete_actions = discrete_actions
        self.network_type = "critic" if self.discrete_actions else "actor"

        self.history_len = history_len
        self.buffer_size = buffer_size
        self.weights_sync_period = weights_sync_period
        self.episode_index = 0
        self.action_clip = action_clip

        self.infer = mode == "infer"
        self.seeds = seeds

        if logdir is not None:
            current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
            logpath = f"{logdir}/sampler-{mode}-{id}-{current_date}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

        self.buffer = SamplerBuffer(
            capacity=self.buffer_size,
            observation_shape=self.env.observation_shape,
            action_shape=self.env.action_shape
        )

        n_atoms = 1 if not self.discrete_actions else self.network.out_features
        self.action_handler = ActionHandler(
            device=self._device,
            discrete_actions=self.discrete_actions,
            deterministic=self.infer,
            critic_distribution=critic_distribution,
            n_atoms=n_atoms,
            values_range=values_range
        )

    def __repr__(self):
        str_val = " ".join(
            [
                f"{key}: {str(getattr(self, key, ''))}"
                for key in ["history_len", "action_clip"]
            ]
        )
        return f"Sampler. {str_val}"

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_network_weights(self):
        if self.resume is not None:
            checkpoint = UtilsFactory.load_checkpoint(self.resume)
            weights = checkpoint[f"{self.network_type}_state_dict"]
            self.network.load_state_dict(weights)
        elif self.redis_server is not None:
            weights = deserialize(
                self.redis_server.get(
                    f"{self.redis_prefix}_{self.network_type}_weights"
                )
            )
            weights = {k: self.to_tensor(v) for k, v in weights.items()}
            self.network.load_state_dict(weights)
        else:
            raise NotImplementedError
        self.network.eval()

    def store_episode(self):
        if self.redis_server is None:
            return
        states, actions, rewards, dones = self.buffer.get_complete_episode()
        episode = [
            states.tolist(),
            actions.tolist(),
            rewards.tolist(),
            dones.tolist()
        ]
        episode = serialize(episode)
        self.redis_server.rpush("trajectories", episode)

    def act(self, state, exploration_strategy):
        with torch.no_grad():
            action = self.action_handler._act_fn(self.network, state)
            if not self.infer:
                action = exploration_strategy._explore(action)
            return action

    def run(self):
        self.episode_index = 1
        self.load_network_weights()
        self.buffer = SamplerBuffer(
            capacity=self.buffer_size,
            observation_shape=self.env.observation_shape,
            action_shape=self.env.action_shape,
            discrete_actions=self.discrete_actions
        )

        seed = self._seed + random.randrange(SEED_RANGE)
        set_global_seeds(seed)
        seed = random.randrange(SEED_RANGE) \
            if self.seeds is None \
            else random.choice(self.seeds)
        set_global_seeds(seed)
        self.buffer.init_with_observation(self.env.reset())

        step_index = 0
        episode_reward = 0
        episode_reward_orig = 0
        start_time = time.time()
        done = False
        states = None

        while True:

            exploration_strategy = self.explorator.get_exploration_strategy()
            if isinstance(exploration_strategy, ParameterSpaceNoise):
                exploration_strategy._run(self.network, states)

            while not done:
                state = self.buffer.get_state(history_len=self.history_len)
                action = self.act(state, exploration_strategy)

                if self.action_clip is not None:
                    action = np.clip(
                        action,
                        a_min=self.action_clip[0],
                        a_max=self.action_clip[1]
                    )

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_reward_orig += info.get("reward_origin", 0)

                transition = [next_state, action, reward, done]
                self.buffer.push_transition(transition)
                step_index += 1

            elapsed_time = time.time() - start_time
            if not self.infer or self.force_store:
                self.store_episode()

            print(
                f"--- episode {self.episode_index:5d}:\t"
                f"steps: {step_index:5d}\t"
                f"reward: {episode_reward:10.4f}/{episode_reward_orig:10.4f}\t"
                f"seed: {seed}"
            )

            if self.logger is not None:
                self.logger.add_scalar("steps", step_index, self.episode_index)
                self.logger.add_scalar(
                    "reward", episode_reward, self.episode_index
                )
                self.logger.add_scalar(
                    "reward_origin", episode_reward_orig, self.episode_index
                )
                self.logger.add_scalar(
                    "episode per minute", 1. / elapsed_time * 60,
                    self.episode_index
                )
                self.logger.add_scalar(
                    "steps per second", step_index / elapsed_time,
                    self.episode_index
                )
                self.logger.add_scalar(
                    "episode time (sec)", elapsed_time, self.episode_index
                )
                self.logger.add_scalar(
                    "episode time (min)", elapsed_time / 60, self.episode_index
                )
                self.logger.add_scalar(
                    "step time (sec)", elapsed_time / step_index,
                    self.episode_index
                )

            self.episode_index += 1

            if self.episode_index >= self.episode_limit:
                return

            if self.episode_index % self.weights_sync_period == 0:
                self.load_network_weights()

                states = self.buffer.get_states_history(
                    history_len=self.history_len
                )
                states = self.to_tensor(states).detach()

            self.buffer = SamplerBuffer(
                capacity=self.buffer_size,
                observation_shape=self.env.observation_shape,
                action_shape=self.env.action_shape,
                discrete_actions=self.discrete_actions
            )

            seed = self._seed + random.randrange(SEED_RANGE)
            set_global_seeds(seed)
            if self.seeds is None:
                seed = random.randrange(SEED_RANGE)
            else:
                seed = random.choice(self.seeds)
            set_global_seeds(seed)
            self.buffer.init_with_observation(self.env.reset())

            step_index = 0
            episode_reward = 0
            episode_reward_orig = 0
            start_time = time.time()
            done = False
