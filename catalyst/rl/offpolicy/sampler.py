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

# speed up optimization
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
SEED_RANGE = 2**32 - 2


def get_actor_weights(actor, exclude_norm=False):
    state_dict = actor.state_dict()
    if exclude_norm:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if all(x not in key for x in ["norm", "lstm"])
        }
    state_dict = {key: value.clone() for key, value in state_dict.items()}
    return state_dict


def set_actor_weights(actor, weights, strict=True):
    actor.load_state_dict(weights, strict=strict)


def set_params_noise(actor, states, target_d=0.2, tol=1e-3, max_steps=1000):
    exclude_norm = True
    orig_weights = get_actor_weights(actor, exclude_norm=exclude_norm)
    orig_act = actor(states)

    sigma_min = 0.
    sigma_max = 100.
    sigma = sigma_max
    step = 0

    while step < max_steps:
        dist = torch.distributions.normal.Normal(0, sigma)
        weights = {
            key: w.clone() + dist.sample(w.shape)
            for key, w in orig_weights.items()
        }
        set_actor_weights(actor, weights, strict=not exclude_norm)
        new_act = actor(states)

        diff = new_act - orig_act
        d = torch.mean(torch.sqrt(torch.sum(torch.pow(diff, 2), 1))).item()

        dd = d - target_d
        if np.abs(dd) < tol:
            break

        # too big sigma
        if dd > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2
        step += 1
    return d


class SamplerBuffer:
    def __init__(self, capacity, observation_shape, action_shape):
        self.size = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.pointer = 0
        self.observations = np.empty(
            (self.size, ) + tuple(self.observation_shape), dtype=np.float32
        )
        self.actions = np.empty(
            (self.size, ) + tuple(self.action_shape), dtype=np.float32
        )
        self.rewards = np.empty((self.size, ), dtype=np.float32)
        self.dones = np.empty((self.size, ), dtype=np.bool)

    def init_with_observation(self, observation):
        self.observations[0] = observation
        self.pointer = 0

    def get_state(self, history_len=1, pointer=None):
        pointer = pointer or self.pointer
        state = np.zeros(
            (history_len, ) + self.observation_shape,
            dtype=np.float32
        )
        indices = np.arange(max(0, pointer - history_len + 1), pointer + 1)
        state[-len(indices):] = self.observations[indices]
        return state

    def push_transition(self, transition):
        """ transition = [s_tp1, a_t, r_t, d_t]
        """
        s_tp1, a_t, r_t, d_t, ts_t = transition
        self.observations[self.pointer + 1] = s_tp1
        self.actions[self.pointer] = a_t
        self.rewards[self.pointer] = r_t
        self.dones[self.pointer] = d_t
        self.pointer += 1

    def get_complete_episode(self):
        indices = np.arange(self.pointer)
        states = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        return states, actions, rewards, dones

    def get_states_history(self, history_len=1):
        states = [
            self.get_state(history_len=history_len, pointer=i)
            for i in range(self.pointer)
        ]
        states = np.array(states)
        return states


class Sampler:
    def __init__(
        self,
        actor,
        env,
        id,
        logdir=None,
        redis_server=None,
        redis_prefix=None,
        buffer_size=int(1e4),
        history_len=1,
        weights_sync_period=1,
        mode="infer",
        resume=None,
        action_noise_prob=0,
        action_noise_t=1,
        random_process=None,
        param_noise_prob=0,
        param_noise_d=0.2,
        param_noise_steps=1000,
        seeds=None,
        action_clip=(-1, 1),
        episode_limit=None,
        force_store=False,
        min_episode_steps=None,
        min_episode_reward=None
    ):

        self._seed = 42 + id
        set_global_seeds(self._seed)

        self._sampler_id = id
        self._device = UtilsFactory.prepare_device()
        self.actor = copy.deepcopy(actor).to(self._device)
        self.env = env
        self.redis_server = redis_server
        self.redis_prefix = redis_prefix or ""
        self.resume = resume
        self.episode_limit = episode_limit or int(2**32 - 2)
        self.force_store = force_store
        self.min_episode_steps = min_episode_steps
        self.min_episode_reward = min_episode_reward
        self.hard_seeds = set()
        min_episode_flag_ = \
            min_episode_steps is None and min_episode_reward is None
        assert min_episode_flag_ or seeds is None

        self.min_episode_steps = self.min_episode_steps or -int(1e6)
        self.min_episode_reward = self.min_episode_reward or -int(1e6)

        self.history_len = history_len
        self.buffer_size = buffer_size
        self.weights_sync_period = weights_sync_period
        self.episode_index = 0
        self.action_clip = action_clip

        self.infer = mode == "infer"
        self.seeds = seeds

        self.action_noise_prob = action_noise_prob
        self.action_noise_t = action_noise_t
        self.random_process = random_process or RandomProcess()

        self.param_noise_prob = param_noise_prob
        self.param_noise_d = param_noise_d
        self.param_noise_steps = param_noise_steps

        if self.infer:
            self.action_noise_prob = 0
            self.param_noise_prob = 0

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

    def __repr__(self):
        str_val = " ".join(
            [
                f"{key}: {str(getattr(self, key, ''))}"
                for key in ["history_len", "action_noise_t", "action_clip"]
            ]
        )
        return f"Sampler. {str_val}"

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def load_actor_weights(self):
        if self.resume is not None:
            checkpoint = UtilsFactory.load_checkpoint(self.resume)
            weights = checkpoint[f"actor_state_dict"]
            self.actor.load_state_dict(weights)
        elif self.redis_server is not None:
            weights = deserialize(
                self.redis_server.get(f"{self.redis_prefix}_actor_weights")
            )
            weights = {k: self.to_tensor(v) for k, v in weights.items()}
            self.actor.load_state_dict(weights)
        else:
            raise NotImplementedError
        self.actor.eval()

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
        hard_seeds = serialize(list(self.hard_seeds))
        self.redis_server.set(
            f"{self.redis_prefix}_{self._sampler_id}_hard_seeds", hard_seeds
        )

    def act(self, state):
        with torch.no_grad():
            states = self.to_tensor(state).unsqueeze(0)
            action = self.actor(states, deterministic=self.infer)
            action = action[0].detach().cpu().numpy()
            return action

    def run(self):
        self.episode_index = 1
        self.load_actor_weights()
        self.buffer = SamplerBuffer(
            self.buffer_size, self.env.observation_shape,
            self.env.action_shape
        )

        seed = self._seed + random.randrange(SEED_RANGE)
        set_global_seeds(seed)
        seed = random.randrange(SEED_RANGE) \
            if self.seeds is None \
            else random.choice(self.seeds)
        set_global_seeds(seed)
        self.buffer.init_with_observation(self.env.reset())
        self.random_process.reset_states()

        action_noise = False
        param_noise_d = 0
        noise_action = 0
        action_noise_t = 0
        step_index = 0
        episode_reward = 0
        episode_reward_orig = 0
        start_time = time.time()
        done = False

        while True:
            while not done:
                state = self.buffer.get_state(history_len=self.history_len)
                action = self.act(state)

                if action_noise \
                        and action_noise_t + self.action_noise_t >= step_index:
                    noise_action = self.random_process.sample()
                    action_noise_t = step_index
                else:
                    noise_action = noise_action

                action = action + noise_action
                action = np.clip(
                    action,
                    a_min=self.action_clip[0],
                    a_max=self.action_clip[1]
                )

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_reward_orig += info.get("reward_origin", 0)

                transition = [next_state, action, reward, done, step_index]
                self.buffer.push_transition(transition)
                step_index += 1

            elapsed_time = time.time() - start_time
            if not self.infer or self.force_store:
                self.store_episode()

            if step_index < self.min_episode_steps \
                    or episode_reward < self.min_episode_reward:
                self.hard_seeds.add(seed)
            else:
                self.hard_seeds.discard(seed)

            print(
                f"--- episode {self.episode_index:5d}:\t"
                f"steps: {step_index:5d}\t"
                f"reward: {episode_reward:10.4f}/{episode_reward_orig:10.4f}\t"
                f"seed: {seed}"
            )

            if self.logger is not None:
                self.logger.add_scalar("steps", step_index, self.episode_index)
                self.logger.add_scalar(
                    "action noise sigma", self.random_process.current_sigma,
                    self.episode_index
                )
                self.logger.add_scalar(
                    "param noise d", param_noise_d, self.episode_index
                )
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
                self.load_actor_weights()

                noise_prob_ = random.random()

                if noise_prob_ < self.param_noise_prob:
                    states = self.buffer.get_states_history(
                        history_len=self.history_len
                    )
                    states = self.to_tensor(states).detach()
                    param_noise_d = set_params_noise(
                        actor=self.actor,
                        states=states,
                        target_d=self.param_noise_d,
                        tol=1e-3,
                        max_steps=self.param_noise_steps
                    )
                    action_noise = False
                elif noise_prob_ < \
                        self.param_noise_prob + self.action_noise_prob:
                    action_noise = True
                    param_noise_d = 0
                else:
                    action_noise = False
                    param_noise_d = 0

            self.buffer = SamplerBuffer(
                capacity=self.buffer_size,
                observation_shape=self.env.observation_shape,
                action_shape=self.env.action_shape
            )

            seed = self._seed + random.randrange(SEED_RANGE)
            set_global_seeds(seed)
            if self.seeds is None:
                hard_seed_prob = random.random()
                if len(self.hard_seeds) > 0 and hard_seed_prob < 0.5:
                    seed = random.sample(self.hard_seeds, 1)[0]
                else:
                    seed = random.randrange(SEED_RANGE)
            else:
                seed = random.choice(self.seeds)
            set_global_seeds(seed)
            self.buffer.init_with_observation(self.env.reset())
            self.random_process.reset_states()

            noise_action = 0
            action_noise_t = 0
            step_index = 0
            episode_reward = 0
            episode_reward_orig = 0
            start_time = time.time()
            done = False
