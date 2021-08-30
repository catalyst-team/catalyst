# flake8: noqa
from typing import Iterator, Optional, Sequence, Tuple
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from catalyst import dl, metrics, utils

# On-policy common

Rollout = namedtuple("Rollout", field_names=["states", "actions", "rewards",])


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, rollout: Rollout):
        self.buffer.append(rollout)

    def sample(self, idx: int) -> Sequence[np.array]:
        states, actions, rewards = self.buffer[idx]
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        return states, actions, rewards

    def __len__(self) -> int:
        return len(self.buffer)


# as far as RL does not have some predefined dataset,
# we need to specify epoch length by ourselfs
class RolloutDataset(IterableDataset):
    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer

    def __iter__(self) -> Iterator[Sequence[np.array]]:
        for i in range(len(self.buffer)):
            states, actions, rewards = self.buffer.sample(i)
            yield states, actions, rewards
        self.buffer.buffer.clear()

    def __len__(self) -> int:
        return self.buffer.capacity


# REINFORCE


def get_cumulative_rewards(rewards, gamma=0.99):
    G = [rewards[-1]]
    for r in reversed(rewards[:-1]):
        G.insert(0, r + gamma * G[0])
    return G


def to_one_hot(y, n_dims=None):
    """ Takes an integer vector and converts it to 1-hot matrix. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def get_action(env, network: nn.Module, state: np.array) -> int:
    state = torch.tensor(state[None], dtype=torch.float32)
    logits = network(state).detach()
    probas = F.softmax(logits, -1).cpu().numpy()[0]
    action = np.random.choice(len(probas), p=probas)
    return int(action)


def generate_session(
    env, network: nn.Module, t_max: int = 1000, rollout_buffer: Optional[RolloutBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    states, actions, rewards = [], [], []
    state = env.reset()

    for t in range(t_max):
        action = get_action(env, network, state=state)
        next_state, reward, done, _ = env.step(action)

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward
        state = next_state
        if done:
            break
    if rollout_buffer is not None:
        rollout_buffer.append(Rollout(states, actions, rewards))

    return total_reward, t


def generate_sessions(
    env,
    network: nn.Module,
    t_max: int = 1000,
    rollout_buffer: Optional[RolloutBuffer] = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env, network=network, t_max=t_max, rollout_buffer=rollout_buffer,
        )
        sessions_reward += r
        sessions_steps += t
    return sessions_reward, sessions_steps


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


# Catalyst


class GameCallback(dl.Callback):
    def __init__(
        self,
        *,
        env,
        rollout_buffer: RolloutBuffer,
        num_train_sessions: int = int(1e2),
        num_valid_sessions: int = int(1e2),
    ):
        super().__init__(order=0)
        self.env = env
        self.rollout_buffer = rollout_buffer
        self.num_train_sessions = num_train_sessions
        self.num_valid_sessions = num_valid_sessions

    def on_epoch_start(self, runner: dl.IRunner):
        self.actor = runner.model

        self.actor.eval()
        train_rewards, train_steps = generate_sessions(
            env=self.env,
            network=self.actor,
            rollout_buffer=self.rollout_buffer,
            num_sessions=self.num_train_sessions,
        )
        train_rewards /= float(self.num_train_sessions)
        train_steps /= float(self.num_train_sessions)
        runner.epoch_metrics["_epoch_"]["t_reward"] = train_rewards
        runner.epoch_metrics["_epoch_"]["t_steps"] = train_steps
        self.actor.train()

    def on_epoch_end(self, runner: dl.IRunner):
        self.actor.eval()
        valid_rewards, valid_steps = generate_sessions(
            env=self.env, network=self.actor, num_sessions=self.num_valid_sessions
        )
        self.actor.train()

        valid_rewards /= float(self.num_valid_sessions)
        valid_steps /= float(self.num_valid_sessions)
        runner.epoch_metrics["_epoch_"]["v_reward"] = valid_rewards
        runner.epoch_metrics["_epoch_"]["v_steps"] = valid_steps


class CustomRunner(dl.Runner):
    def __init__(
        self, *, gamma: float, entropy_coef: float = 0.1, **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma: float = gamma
        self.entropy_coef: float = entropy_coef

    def on_loader_start(self, runner: dl.IRunner):
        super().on_loader_start(runner)
        self.meters = {key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss"]}

    def handle_batch(self, batch: Sequence[np.array]):
        # model train/valid step
        # ATTENTION:
        #   because of different trajectories lens
        #   ONLY batch_size==1 supported
        states, actions, rewards = batch
        states, actions, rewards = states[0], actions[0], rewards[0]
        cumulative_returns = torch.tensor(get_cumulative_rewards(rewards, gamma))

        logits = self.model(states)
        probas = F.softmax(logits, -1)
        logprobas = F.log_softmax(logits, -1)
        n_actions = probas.shape[1]
        logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions, n_dims=n_actions), dim=1)

        J_hat = torch.mean(logprobas_for_actions * cumulative_returns)
        entropy_reg = -torch.mean(torch.sum(probas * logprobas, dim=1))
        loss = -J_hat - self.entropy_coef * entropy_reg

        self.batch_metrics.update({"loss": loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner: dl.IRunner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


if __name__ == "__main__":
    batch_size = 1
    epoch_size = int(1e3) * batch_size
    buffer_size = int(1e2)
    # runner settings
    gamma = 0.99
    # optimization
    lr = 3e-4

    # env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    rollout_buffer = RolloutBuffer(buffer_size)

    model = get_network(env)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loaders = {
        "train_game": DataLoader(RolloutDataset(rollout_buffer), batch_size=batch_size,),
    }

    runner = CustomRunner(gamma=gamma)
    runner.train(
        engine=dl.DeviceEngine("cpu"),  # for simplicity reasons, let's run everything on cpu
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs_dqn",
        num_epochs=10,
        verbose=True,
        valid_loader="_epoch_",
        valid_metric="v_reward",
        minimize_valid_metric=False,
        load_best_on_end=True,
        callbacks=[GameCallback(env=env, rollout_buffer=rollout_buffer,)],
    )

    env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_reinforce", force=True)
    generate_sessions(env=env, network=model, num_sessions=100)
    env.close()

    # from IPython.display import HTML
    # import os
    #
    # video_names = list(filter(lambda s: s.endswith(".mp4"), os.listdir("./videos_reinforce/")))
    #
    # HTML("""
    # <video width="640" height="480" controls>
    #   <source src="{}" type="video/mp4">
    # </video>
    # """.format("./videos/" + video_names[-1]))
    # # this may or may not be _last_ video. Try other indices
