#!/usr/bin/env python

import gym
from .environment import EnvironmentWrapper


class GymEnvWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env_name,  # ["LunarLander-v2", "LunarLanderContinuous-v2"]
        **params
    ):
        env = gym.make(env_name)
        super().__init__(env=env, **params)


__all__ = ["GymEnvWrapper"]
