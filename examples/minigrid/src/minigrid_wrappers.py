import cv2

import gym
from gym import spaces
import gym_minigrid  # noqa: F401

cv2.ocl.setUseOpenCL(False)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super().__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super().__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        img_space: spaces.Box = self.observation_space.spaces["image"]

        self.observation_space = spaces.Dict(
            {
                "image": gym.spaces.Box(
                    img_space.low[0, 0, 0],
                    img_space.high[0, 0, 0], [
                        img_space.shape[self.op[0]],
                        img_space.shape[self.op[1]], img_space.shape[self.op[2]
                                                                     ]
                    ],
                    dtype=self.observation_space.dtype
                )
            }
        )

    def observation(self, ob):
        ob = {
            "image": ob["image"].transpose(self.op[0], self.op[1], self.op[2])
        }
        return ob


def make_minigrid_env(env_id):
    env = gym.make(env_id)
    env = TransposeImage(env, op=[2, 0, 1])
    return env
