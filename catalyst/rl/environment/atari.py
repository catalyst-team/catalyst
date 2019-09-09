#!/usr/bin/env python

from .environment import EnvironmentWrapper

from .env_wrappers import make_atari_env


class AtariEnvWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env_id,
        max_episode_steps=None,
        episode_life=True,
        clip_rewards=False,
        width=84,
        height=84,
        grayscale=True,
        **params
    ):
        env = make_atari_env(
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            episode_life=episode_life,
            clip_rewards=clip_rewards,
            width=width,
            height=height,
            grayscale=grayscale,
        )
        super().__init__(env=env, **params)


__all__ = ["AtariEnvWrapper"]
