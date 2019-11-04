#!/usr/bin/env python

from catalyst.rl.environment import EnvironmentWrapper
from .minigrid_wrappers import make_minigrid_env


class MiniGridEnvWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env_id,
        **params
    ):
        env = make_minigrid_env(env_id)
        super().__init__(env=env, **params)
