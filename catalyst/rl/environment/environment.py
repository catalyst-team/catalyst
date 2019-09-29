#!/usr/bin/env python

from typing import Union, List
import time
import random

import numpy as np
from gym import spaces

from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space


class EnvironmentWrapper(EnvironmentSpec):
    def __init__(
        self,
        env,
        history_len: int = 1,
        frame_skip: Union[int, List] = 1,
        reward_scale: float = 1,
        step_reward: float = 0.0,
        max_fps: int = None,
        observation_mean: np.ndarray = None,
        observation_std: np.ndarray = None,
        action_mean: np.ndarray = None,
        action_std: np.ndarray = None,
        visualize: bool = False,
        mode: str = "train",
        sampler_id: int = None,
        use_virtual_display: bool = False
    ):
        super().__init__(visualize=visualize, mode=mode, sampler_id=sampler_id)

        if use_virtual_display:
            # virtual display hack
            from pyvirtualdisplay import Display
            from pyvirtualdisplay.randomize import Randomizer
            self.display = Display(
                visible=0, size=(1366, 768), randomizer=Randomizer())
            self.display.start()

        self.env = env

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._reward_scale = reward_scale
        self._step_reward = step_reward
        self._max_fps = max_fps if self._mode == "train" else None
        self._min_delay_between_steps = 1. / self._max_fps \
            if self._max_fps is not None \
            else 0
        self._last_step_time = time.time()

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
        self._last_step_time = time.time()
        observation = self.env.reset()
        observation = self._process_observation(observation)
        return observation

    def step(self, action):
        delay_between_steps = time.time() - self._last_step_time
        time.sleep(max(0, self._min_delay_between_steps - delay_between_steps))
        self._last_step_time = time.time()

        reward, raw_reward = 0, 0
        action = self._process_action(action)
        frame_skip = self._frame_skip \
            if isinstance(self._frame_skip, int) \
            else random.randint(self._frame_skip[0], self._frame_skip[1])
        for i in range(frame_skip):
            observation, r, done, info = self.env.step(action)
            if self._visualize:
                self.env.render()
            reward += r + self._step_reward
            info = info or {}
            raw_reward += info.get("raw_reward", r)
            if done:
                break
        info = info or {}
        info["raw_reward"] = raw_reward
        reward *= self._reward_scale
        observation = self._process_observation(observation)
        return observation, reward, done, info
