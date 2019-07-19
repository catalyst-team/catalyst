#!/usr/bin/env python

import time
import numpy as np
import gym
from gym import spaces
from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space


class GymWrapper(EnvironmentSpec):
    def __init__(
        self,
        env_name,  # ["LunarLander-v2", "LunarLanderContinuous-v2"]
        # env_wrappers=None,
        history_len=1,
        frame_skip=1,
        reward_scale=1,
        step_delay=0.0,
        observation_mean=None,
        observation_std=None,
        action_mean=None,
        action_std=None,
        visualize=False,
        mode="train",
    ):
        super().__init__(visualize=visualize, mode=mode)

        self.env = gym.make(env_name)
        # @TODO: add logic with registry and env_wrappers

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._visualize = visualize
        self._reward_scale = reward_scale
        self._step_delay = step_delay

        self.observation_mean = np.array(observation_mean) \
            if observation_mean is not None else None
        self.observation_std = np.array(observation_std) \
            if observation_std is not None else None
        self.action_mean = np.array(action_mean) \
            if action_mean is not None else None
        self.action_std = np.array(action_std) \
            if action_std is not None else None

        self._prepare_spaces()

    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> spaces.space.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.space.Space:
        return self._state_space

    @property
    def action_space(self) -> spaces.space.Space:
        return self._action_space

    def _prepare_spaces(self):
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

        self._state_space = extend_space(
            self._observation_space, self._history_len
        )

    def _process_observation(self, observation):
        if self.observation_mean is not None \
                and self.observation_std is not None:
            observation = \
                (observation - self.observation_mean) \
                / (self.observation_std + 1e-8)
        return observation

    def _process_action(self, action):
        if self.action_mean is not None \
                and self.action_std is not None:
            action = action * (self.action_std + 1e-8) + self.action_mean
        return action

    def reset(self):
        observation = self.env.reset()
        observation = self._process_observation(observation)
        return observation

    def step(self, action):
        time.sleep(self._step_delay)
        reward = 0
        action = self._process_action(action)
        for i in range(self._frame_skip):
            observation, r, done, info = self.env.step(action)
            if self._visualize:
                self.env.render()
            reward += r
            if done:
                break
        info["raw_reward"] = reward
        reward *= self._reward_scale
        observation = self._process_observation(observation)
        return observation, reward, done, info


__all__ = ["GymWrapper"]
