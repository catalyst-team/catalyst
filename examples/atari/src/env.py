#!/usr/bin/env python

from catalyst.rl.environment import EnvironmentWrapper

from .atari_wrappers import make_atari_env


class AtariEnvWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env_id,
        max_episode_steps=None,
        episode_life=True,
        clip_rewards=False,
        frame_stack=False,
        scale=False,
        **params
    ):
        env = make_atari_env(
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            episode_life=episode_life,
            clip_rewards=clip_rewards,
            frame_stack=frame_stack,
            scale=scale,
        )
        super().__init__(env=env, **params)
