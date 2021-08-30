# flake8: noqa
from typing import Optional, Sequence
import os

from buffer import OffpolicyReplayBuffer
from db import RedisDB
import gym
from misc import GameCallback, soft_update, Trajectory
import numpy as np
from sampler import ISampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from catalyst import dl, metrics, utils

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


class Sampler(ISampler):
    def get_action(
        self, env, network: nn.Module, state: np.array, sigma: Optional[float] = None
    ) -> np.array:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = network(state).detach().cpu().numpy()[0]
        if sigma is not None:
            action = np.random.normal(action, sigma)
        return action

    def get_trajectory(
        self,
        env: gym.Env,
        actor: nn.Module,
        device,
        sampler_index: int = None,
        trajectory_index: int = None,
    ) -> Trajectory:
        if sampler_index is not None:
            sigma = float(0.2 * pow(0.9996, trajectory_index + 1) / (sampler_index + 1))
        else:
            sigma = None
        state = env.reset()
        observations, actions, rewards, dones = [], [], [], []

        for t in range(env.spec.max_episode_steps):
            action = self.get_action(env, actor, state=state, sigma=sigma)
            next_state, reward, done, _ = env.step(action)

            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                break

        trajectory = Trajectory(observations, actions, rewards, dones)
        return trajectory


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
        # states, actions, rewards, dones, next_states = batch
        states, actions, rewards, next_states, dones = (
            batch["state"].squeeze_(1).to(torch.float32),
            batch["action"].to(torch.float32),
            batch["reward"].to(torch.float32),
            batch["next_state"].squeeze_(1).to(torch.float32),
            batch["done"].to(torch.bool),
        )

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
    num_samplers = 2
    batch_size = 256
    epoch_size = int(1e2) * batch_size
    buffer_size = int(1e5)
    # runner settings, ~training
    gamma = 0.99
    tau = 0.01
    tau_period = 1
    # optimization
    lr_actor = 1e-4
    lr_critic = 1e-3

    db_server = RedisDB()

    # You can change game
    # env_name = "LunarLanderContinuous-v2"
    env_name = "Pendulum-v0"
    env = NormalizedActions(gym.make(env_name))

    replay_buffer = OffpolicyReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        epoch_len=epoch_size,
        capacity=buffer_size,
        n_step=1,
        gamma=gamma,
        history_len=1,
    )

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

    loaders = {"train_game": DataLoader(replay_buffer, batch_size=batch_size)}

    runner = CustomRunner(gamma=gamma, tau=tau, tau_period=tau_period,)

    runner.train(
        # for simplicity reasons, let's run everything on single gpu
        engine=dl.DeviceEngine("cuda"),
        model=models,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs_ddpg",
        num_epochs=50,
        verbose=True,
        valid_loader="_epoch_",
        valid_metric="reward",
        minimize_valid_metric=False,
        load_best_on_end=True,
        callbacks=[
            GameCallback(
                sampler_fn=Sampler,
                env=env,
                replay_buffer=replay_buffer,
                db_server=db_server,
                actor_key="actor",
                num_samplers=num_samplers,
                min_transactions_num=epoch_size,
            )
        ],
    )

    # env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_ddpg", force=True)
    # generate_sessions(env=env, network=runner.model["actor"], num_sessions=100)
    # env.close()

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
