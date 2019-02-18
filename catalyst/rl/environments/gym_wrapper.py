#!/usr/bin/env python

import gym
import time


class GymWrapper:
    def __init__(
        self,
        env_name="LunarLanderContinuous-v2",
        frame_skip=1,
        visualize=False,
        reward_scale=1,
        step_delay=0.1
    ):
        self.env = gym.make(env_name)

        self.visualize = visualize
        self.frame_skip = frame_skip
        self.reward_scale = reward_scale
        self.step_delay = step_delay

        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        self.time_step = 0
        self.total_reward = 0

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        return self.env.reset()

    def step(self, action):
        time.sleep(self.step_delay)
        reward = 0
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            if self.visualize:
                self.env.render()
            reward += r
            if done:
                break
        self.total_reward += reward
        self.time_step += 1
        info["reward_origin"] = reward
        reward *= self.reward_scale
        return observation, reward, done, info
