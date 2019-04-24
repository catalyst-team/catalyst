#!/usr/bin/env python

import numpy as np
import gym
from gym import spaces
import time
from .core import EnvironmentSpec


class GymWrapper(EnvironmentSpec):
    def __init__(
        self,
        env_name="LunarLander-v2",  # "LunarLanderContinuous-v2",
        # env_wrappers=None,
        history_len=1,
        frame_skip=1,
        visualize=False,
        reward_scale=1,
        step_delay=0.0
    ):
        self.env = gym.make(env_name)
        # @TODO: add logic with registry and env_wrappers

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._visualize = visualize
        self._reward_scale = reward_scale
        self._step_delay = step_delay

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

        def _expand_to_history_len(np_array):
            return np.concatenate(
                self._history_len * [np.expand_dims(np_array, 0)], axis=0)

        if isinstance(self._observation_space, spaces.Box):
            self._state_space = spaces.Box(
                low=_expand_to_history_len(self._observation_space.low),
                high=_expand_to_history_len(self._observation_space.high),
                # shape=(self._history_len,) + self._observation_space.shape,
                dtype=self._observation_space.dtype
            )
        else:
            raise NotImplementedError("not yet implemented")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        time.sleep(self._step_delay)
        reward = 0
        for i in range(self._frame_skip):
            observation, r, done, info = self.env.step(action)
            if self._visualize:
                self.env.render()
            reward += r
            if done:
                break
        info["reward_origin"] = reward
        reward *= self._reward_scale
        return observation, reward, done, info
