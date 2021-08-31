# flake8: noqa
from typing import Iterator, Optional, Sequence, Tuple
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from catalyst import dl, metrics, utils

# Off-policy common

Transition = namedtuple(
    "Transition", field_names=["state", "action", "reward", "done", "next_state"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, size: int) -> Sequence[np.array]:
        indices = np.random.choice(len(self.buffer), size, replace=size > len(self.buffer))
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        next_states = np.array(next_states, dtype=np.float32)
        return states, actions, rewards, dones, next_states

    def __len__(self) -> int:
        return len(self.buffer)


# as far as RL does not have some predefined dataset,
# we need to specify epoch length by ourselfs
class ReplayDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, epoch_size: int = int(1e3)):
        self.buffer = buffer
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[Sequence[np.array]]:
        states, actions, rewards, dones, next_states = self.buffer.sample(self.epoch_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], next_states[i]

    def __len__(self) -> int:
        return self.epoch_size


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Updates the `target` data with the `source` one smoothing by ``tau`` (inplace operation)."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# DDPG


class NormalizedActions(gym.ActionWrapper):
    def action(self, action: float) -> float:
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action: float) -> float:
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


def get_action(
    env, network: nn.Module, state: np.array, sigma: Optional[float] = None
) -> np.array:
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = network(state).detach().cpu().numpy()[0]
    if sigma is not None:
        action = np.random.normal(action, sigma)
    return action


def generate_session(
    env,
    network: nn.Module,
    sigma: Optional[float] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    state = env.reset()

    for t in range(env.spec.max_episode_steps):
        action = get_action(env, network, state=state, sigma=sigma)
        next_state, reward, done, _ = env.step(action)

        if replay_buffer is not None:
            transition = Transition(state, action, reward, done, next_state)
            replay_buffer.append(transition)

        total_reward += reward
        state = next_state
        if done:
            break

    return total_reward, t


def generate_sessions(
    env,
    network: nn.Module,
    sigma: Optional[float] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env, network=network, sigma=sigma, replay_buffer=replay_buffer,
        )
        sessions_reward += r
        sessions_steps += t
    return sessions_reward, sessions_steps


def get_network_actor(env):
    inner_fn = utils.get_optimal_inner_init(nn.ReLU)
    outer_fn = utils.outer_init

    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(),
    )
    head = torch.nn.Sequential(nn.Linear(300, 1), nn.Tanh())

    network.apply(inner_fn)
    head.apply(outer_fn)

    return torch.nn.Sequential(network, head)


def get_network_critic(env):
    inner_fn = utils.get_optimal_inner_init(nn.LeakyReLU)
    outer_fn = utils.outer_init

    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0] + 1, 400),
        nn.LeakyReLU(0.01),
        nn.Linear(400, 300),
        nn.LeakyReLU(0.01),
    )
    head = nn.Linear(300, 1)

    network.apply(inner_fn)
    head.apply(outer_fn)

    return torch.nn.Sequential(network, head)


# Catalyst


class GameCallback(dl.Callback):
    def __init__(
        self,
        *,
        env,
        replay_buffer: ReplayBuffer,
        session_period: int,
        sigma: float,
        actor_key: str,
        num_start_sessions: int = int(1e3),
        num_valid_sessions: int = int(1e2),
    ):
        super().__init__(order=0)
        self.env = env
        self.replay_buffer = replay_buffer
        self.session_period = session_period
        self.sigma = sigma
        self.actor_key = actor_key
        self.num_start_sessions = num_start_sessions
        self.num_valid_sessions = num_valid_sessions
        self.session_counter = 0
        self.session_steps = 0

    def on_stage_start(self, runner: dl.IRunner):
        self.actor = runner.model[self.actor_key]

        self.actor.eval()
        generate_sessions(
            env=self.env,
            network=self.actor,
            sigma=self.sigma,
            replay_buffer=self.replay_buffer,
            num_sessions=self.num_start_sessions,
        )
        self.actor.train()

    def on_epoch_start(self, runner: dl.IRunner):
        self.session_counter = 0
        self.session_steps = 0

    def on_batch_end(self, runner: dl.IRunner):
        if runner.global_batch_step % self.session_period == 0:
            self.actor.eval()

            session_reward, session_steps = generate_session(
                env=self.env,
                network=self.actor,
                sigma=self.sigma,
                replay_buffer=self.replay_buffer,
            )

            self.session_counter += 1
            self.session_steps += session_steps

            runner.batch_metrics.update({"s_reward": session_reward})
            runner.batch_metrics.update({"s_steps": session_steps})

            self.actor.train()

    def on_epoch_end(self, runner: dl.IRunner):
        self.actor.eval()
        valid_rewards, valid_steps = generate_sessions(
            env=self.env, network=self.actor, num_sessions=int(self.num_valid_sessions)
        )
        self.actor.train()

        valid_rewards /= float(self.num_valid_sessions)
        valid_steps /= float(self.num_valid_sessions)
        runner.epoch_metrics["_epoch_"]["num_sessions"] = self.session_counter
        runner.epoch_metrics["_epoch_"]["num_samples"] = self.session_steps
        runner.epoch_metrics["_epoch_"]["updates_per_sample"] = (
            runner.loader_sample_step / self.session_steps
        )
        runner.epoch_metrics["_epoch_"]["v_reward"] = valid_rewards
        runner.epoch_metrics["_epoch_"]["v_steps"] = valid_steps


class CustomRunner(dl.Runner):
    def __init__(
        self,
        *,
        gamma: float,
        tau: float,
        tau_period: int = 1,
        actor_key: str = "actor",
        critic_key: str = "critic",
        target_actor_key: str = "target_actor",
        target_critic_key: str = "target_critic",
        actor_optimizer_key: str = "actor",
        critic_optimizer_key: str = "critic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.tau_period = tau_period
        self.actor_key: str = actor_key
        self.critic_key: str = critic_key
        self.target_actor_key: str = target_actor_key
        self.target_critic_key: str = target_critic_key
        self.actor_optimizer_key: str = actor_optimizer_key
        self.critic_optimizer_key: str = critic_optimizer_key
        self.actor: nn.Module = None
        self.critic: nn.Module = None
        self.target_actor: nn.Module = None
        self.target_critic: nn.Module = None
        self.actor_optimizer: nn.Module = None
        self.critic_optimizer: nn.Module = None

    def on_stage_start(self, runner: dl.IRunner):
        super().on_stage_start(runner)
        self.actor = self.model[self.actor_key]
        self.critic = self.model[self.critic_key]
        self.target_actor = self.model[self.target_actor_key]
        self.target_critic = self.model[self.target_critic_key]
        soft_update(self.target_actor, self.actor, 1.0)
        soft_update(self.target_critic, self.critic, 1.0)
        self.actor_optimizer = self.optimizer[self.actor_optimizer_key]
        self.critic_optimizer = self.optimizer[self.critic_optimizer_key]

    def on_loader_start(self, runner: dl.IRunner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["critic_loss", "actor_loss"]
        }

    def handle_batch(self, batch: Sequence[torch.Tensor]):
        # model train/valid step
        states, actions, rewards, dones, next_states = batch

        # get actions for the current state
        pred_actions = self.actor(states)
        # get q-values for the actions in current states
        pred_critic_states = torch.cat([states, pred_actions], 1)
        # use q-values to train the actor model
        policy_loss = (-self.critic(pred_critic_states)).mean()

        with torch.no_grad():
            # get possible actions for the next states
            next_state_actions = self.target_actor(next_states)
            # get possible q-values for the next actions
            next_critic_states = torch.cat([next_states, next_state_actions], 1)
            next_state_values = self.target_critic(next_critic_states).detach().squeeze()
            next_state_values[dones] = 0.0

        # compute Bellman's equation value
        target_state_values = next_state_values * self.gamma + rewards
        # compute predicted values
        critic_states = torch.cat([states, actions], 1)
        state_values = self.critic(critic_states).squeeze()

        # train the critic model
        value_loss = self.criterion(state_values, target_state_values.detach())

        self.batch_metrics.update({"critic_loss": value_loss, "actor_loss": policy_loss})
        for key in ["critic_loss", "actor_loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            self.actor.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic.zero_grad()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            if self.global_batch_step % self.tau_period == 0:
                soft_update(self.target_actor, self.actor, self.tau)
                soft_update(self.target_critic, self.critic, self.tau)

    def on_loader_end(self, runner: dl.IRunner):
        for key in ["critic_loss", "actor_loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


if __name__ == "__main__":
    # data
    batch_size = 64
    epoch_size = int(1e3) * batch_size
    buffer_size = int(1e5)
    # runner settings, ~training
    gamma = 0.99
    tau = 0.01
    tau_period = 1
    # callback, ~exploration
    session_period = 1
    sigma = 0.3
    # optimization
    lr_actor = 1e-4
    lr_critic = 1e-3

    # You can change game
    # env_name = "LunarLanderContinuous-v2"
    env_name = "Pendulum-v0"
    env = NormalizedActions(gym.make(env_name))
    replay_buffer = ReplayBuffer(buffer_size)

    actor, target_actor = get_network_actor(env), get_network_actor(env)
    critic, target_critic = get_network_critic(env), get_network_critic(env)
    utils.set_requires_grad(target_actor, requires_grad=False)
    utils.set_requires_grad(target_critic, requires_grad=False)

    models = {
        "actor": actor,
        "critic": critic,
        "target_actor": target_actor,
        "target_critic": target_critic,
    }

    criterion = torch.nn.MSELoss()
    optimizer = {
        "actor": torch.optim.Adam(actor.parameters(), lr_actor),
        "critic": torch.optim.Adam(critic.parameters(), lr=lr_critic),
    }

    loaders = {
        "train_game": DataLoader(
            ReplayDataset(replay_buffer, epoch_size=epoch_size), batch_size=batch_size,
        ),
    }

    runner = CustomRunner(gamma=gamma, tau=tau, tau_period=tau_period,)

    runner.train(
        engine=dl.DeviceEngine("cpu"),  # for simplicity reasons, let's run everything on cpu
        model=models,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs_ddpg",
        num_epochs=10,
        verbose=True,
        valid_loader="_epoch_",
        valid_metric="v_reward",
        minimize_valid_metric=False,
        load_best_on_end=True,
        callbacks=[
            GameCallback(
                env=env,
                replay_buffer=replay_buffer,
                session_period=session_period,
                sigma=sigma,
                actor_key="actor",
            )
        ],
    )

    env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_ddpg", force=True)
    generate_sessions(env=env, network=runner.model["actor"], num_sessions=100)
    env.close()

    # # show video
    # from IPython.display import HTML
    # import os
    #
    # video_names = list(filter(lambda s: s.endswith(".mp4"), os.listdir("./videos_ddpg/")))
    #
    # HTML("""
    # <video width="640" height="480" controls>
    #   <source src="{}" type="video/mp4">
    # </video>
    # """.format("./videos/" + video_names[-1]))
    # # this may or may not be _last_ video. Try other indices
