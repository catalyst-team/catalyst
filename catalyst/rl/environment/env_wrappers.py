import numpy as np
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super().__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        self._dict_env_space = isinstance(env.observation_space, spaces.Dict)

        img_space: spaces.Box = env.observation_space.spaces["image"] \
            if self._dict_env_space \
            else env.observation_space

        self.observation_space = gym.spaces.Box(
            low=img_space.low[0, 0, 0],
            high=img_space.high[0, 0, 0],
            shape=[
                img_space.shape[self.op[0]],
                img_space.shape[self.op[1]],
                img_space.shape[self.op[2]],
            ],
            dtype=img_space.dtype
        )

    def observation(self, observation):
        if self._dict_env_space:
            observation = observation["image"]
        observation = observation.transpose(self.op[0], self.op[1], self.op[2])
        return observation


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0
            # condition for a few frames
            # so it"s important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 1),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


def make_image_env(env_id):
    env = gym.make(env_id)
    env = TransposeImage(env, op=[2, 0, 1])
    return env


def make_atari_env(
    env_id,
    max_episode_steps=None,
    episode_life=True,
    clip_rewards=False,
    width=84,
    height=84,
    grayscale=True,
):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, height=height, width=width, grayscale=grayscale)
    if clip_rewards:
        env = ClipRewardEnv(env)

    env = TransposeImage(env, op=[2, 0, 1])
    return env
