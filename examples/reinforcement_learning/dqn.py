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


# DQN


def get_action(env, network: nn.Module, state: np.array, epsilon: float = -1) -> int:
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.tensor(state[None], dtype=torch.float32)
        q_values = network(state).detach().cpu().numpy()[0]
        action = np.argmax(q_values)

    return int(action)


def generate_session(
    env,
    network: nn.Module,
    t_max: int = 1000,
    epsilon: float = -1,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    state = env.reset()

    for t in range(t_max):
        action = get_action(env, network, state=state, epsilon=epsilon)
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
    t_max: int = 1000,
    epsilon: float = -1,
    replay_buffer: ReplayBuffer = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env, network=network, t_max=t_max, epsilon=epsilon, replay_buffer=replay_buffer,
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
        replay_buffer: ReplayBuffer,
        session_period: int,
        epsilon: float,
        epsilon_k: float,
        actor_key: str,
        num_start_sessions: int = int(1e3),
        num_valid_sessions: int = int(1e2),
    ):
        super().__init__(order=0)
        self.env = env
        self.replay_buffer = replay_buffer
        self.session_period = session_period
        self.epsilon = epsilon
        self.epsilon_k = epsilon_k
        self.actor_key = actor_key
        self.actor: nn.Module = None
        self.num_start_sessions = num_start_sessions
        self.num_valid_sessions = num_valid_sessions
        self.session_counter = 0
        self.session_steps = 0

    def on_stage_start(self, runner: dl.IRunner) -> None:
        self.actor = runner.model[self.actor_key]

        self.actor.eval()
        generate_sessions(
            env=self.env,
            network=self.actor,
            epsilon=self.epsilon,
            replay_buffer=self.replay_buffer,
            num_sessions=self.num_start_sessions,
        )
        self.actor.train()

    def on_epoch_start(self, runner: dl.IRunner):
        self.epsilon *= self.epsilon_k
        self.session_counter = 0
        self.session_steps = 0

    def on_batch_end(self, runner: dl.IRunner):
        if runner.global_batch_step % self.session_period == 0:
            self.actor.eval()

            session_reward, session_steps = generate_session(
                env=self.env,
                network=self.actor,
                epsilon=self.epsilon,
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
        runner.epoch_metrics["_epoch_"]["epsilon"] = self.epsilon
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
        states, actions, rewards, dones, next_states = batch

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
    batch_size = 64
    epoch_size = int(1e3) * batch_size
    buffer_size = int(1e5)
    # runner settings, ~training
    gamma = 0.99
    tau = 0.01
    tau_period = 1  # in batches
    # callback, ~exploration
    session_period = 100  # in batches
    epsilon = 0.98
    epsilon_k = 0.9
    # optimization
    lr = 3e-4

    # env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)

    network, target_network = get_network(env), get_network(env)
    utils.set_requires_grad(target_network, requires_grad=False)
    models = {"origin": network, "target": target_network}
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loaders = {
        "train_game": DataLoader(
            ReplayDataset(replay_buffer, epoch_size=epoch_size), batch_size=batch_size,
        ),
    }

    runner = CustomRunner(gamma=gamma, tau=tau, tau_period=tau_period)
    runner.train(
        engine=dl.DeviceEngine("cpu"),  # for simplicity reasons, let's run everything on cpu
        model=models,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs_dqn",
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
                epsilon=epsilon,
                epsilon_k=epsilon_k,
                actor_key="origin",
            )
        ],
    )

    env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_dqn", force=True)
    generate_sessions(env=env, network=runner.model["origin"], num_sessions=100)
    env.close()

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
