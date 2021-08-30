# flake8: noqa
from typing import Sequence
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


# DQN
class Sampler(ISampler):
    def get_action(self, env, actor: nn.Module, state: np.array, epsilon: float = -1) -> int:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(state[None], dtype=torch.float32)
            q_values = actor(state).detach().cpu().numpy()[0]
            action = np.argmax(q_values)

        return int(action)

    def get_trajectory(
        self,
        env: gym.Env,
        actor: nn.Module,
        device,
        sampler_index: int = None,
        trajectory_index: int = None,
        t_max: int = 1000,
    ) -> Trajectory:
        if sampler_index is not None:
            epsilon = float(pow(0.9996, trajectory_index + 1) / (sampler_index + 1))
        else:
            epsilon = None
        state = env.reset()
        observations, actions, rewards, dones = [], [], [], []

        for t in range(t_max):
            action = self.get_action(env, actor, state=state, epsilon=epsilon)
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


def get_network(env, num_hidden: int = 128):
    inner_fn = utils.get_optimal_inner_init(nn.ReLU)
    outer_fn = utils.outer_init

    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_hidden),
        nn.ReLU(),
    )
    head = nn.Linear(num_hidden, env.action_space.n)

    network.apply(inner_fn)
    head.apply(outer_fn)

    return torch.nn.Sequential(network, head)


# Catalyst.RL


class CustomRunner(dl.Runner):
    def __init__(
        self,
        *,
        gamma: float,
        tau: float,
        tau_period: int = 1,
        origin_key: str = "origin",
        target_key: str = "target",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma: float = gamma
        self.tau: float = tau
        self.tau_period: int = tau_period
        self.origin_key: str = origin_key
        self.target_key: str = target_key
        self.origin_network: nn.Module = None
        self.target_network: nn.Module = None

    def on_stage_start(self, runner: dl.IRunner):
        super().on_stage_start(runner)
        self.origin_network = self.model[self.origin_key]
        self.target_network = self.model[self.target_key]
        soft_update(self.target_network, self.origin_network, 1.0)

    def on_loader_start(self, runner: dl.IRunner):
        super().on_loader_start(runner)
        self.meters = {key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss"]}

    def handle_batch(self, batch: Sequence[np.array]):
        # model train/valid step
        states, actions, rewards, next_states, dones = (
            batch["state"].squeeze_(1).to(torch.float32),
            batch["action"].to(torch.int64),
            batch["reward"].to(torch.float32),
            batch["next_state"].squeeze_(1).to(torch.float32),
            batch["done"].to(torch.bool),
        )

        # get q-values for all actions in current states
        state_qvalues = self.origin_network(states)
        # select q-values for chosen actions
        state_action_qvalues = state_qvalues.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # compute q-values for all actions in next states
        # compute V*(next_states) using predicted next q-values
        # at the last state we shall use simplified formula:
        # Q(s,a) = r(s,a) since s' doesn't exist
        with torch.no_grad():
            next_state_qvalues = self.target_network(next_states)
            next_state_values = next_state_qvalues.max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        # compute "target q-values" for loss,
        # it's what's inside square parentheses in the above formula.
        target_state_action_qvalues = next_state_values * self.gamma + rewards

        # mean squared error loss to minimize
        loss = self.criterion(state_action_qvalues, target_state_action_qvalues.detach())
        self.batch_metrics.update({"loss": loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.global_batch_step % self.tau_period == 0:
                soft_update(self.target_network, self.origin_network, self.tau)

    def on_loader_end(self, runner: dl.IRunner):
        for key in ["loss"]:
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
    tau_period = 1  # in batches
    # optimization
    lr = 3e-4

    db_server = RedisDB()

    # You can change game
    # env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    replay_buffer = OffpolicyReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        epoch_len=epoch_size,
        capacity=buffer_size,
        n_step=1,
        gamma=gamma,
        history_len=1,
    )

    network, target_network = get_network(env), get_network(env)
    utils.set_requires_grad(target_network, requires_grad=False)
    models = {"origin": network, "target": target_network}
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loaders = {"train_game": DataLoader(replay_buffer, batch_size=batch_size)}

    runner = CustomRunner(gamma=gamma, tau=tau, tau_period=tau_period)
    runner.train(
        # for simplicity reasons, let's run everything on single gpu
        engine=dl.DeviceEngine("cuda"),
        model=models,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs_dqn",
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
                actor_key="origin",
                num_samplers=num_samplers,
                min_transactions_num=epoch_size,
            )
        ],
    )

    # env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_dqn", force=True)
    # generate_sessions(env=env, network=runner.model["origin"], num_sessions=100)
    # env.close()

    # # show video
    # from IPython.display import HTML
    # import os
    #
    # video_names = list(filter(lambda s: s.endswith(".mp4"), os.listdir("./videos_dqn/")))
    #
    # HTML("""
    # <video width="640" height="480" controls>
    #   <source src="{}" type="video/mp4">
    # </video>
    # """.format("./videos/" + video_names[-1]))
    # # this may or may not be _last_ video. Try other indices
