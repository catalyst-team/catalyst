import logging

import numpy as np

import gym
from gym.envs.registration import register

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_ENABLED = True
except ImportError:
    logger.warning(
        "pygame not available, disabling rendering. "
        "To install pygame, run `pip install pygame`."
    )
    PYGAME_ENABLED = False

BRIGHT_COLOR = (200, 200, 200)
DARK_COLOR = (150, 150, 150)


class PointEnv(gym.Env):
    def __init__(
        self,
        goal=(0, 0),
        random_start=False,
        completion_bonus=0.,
        action_scale=0.1,
        max_steps=200
    ):
        self._goal = np.array(goal, dtype=np.float32)
        self._point = np.zeros(2)
        self._completion_bonus = completion_bonus
        self._action_scale = action_scale
        self._max_steps = max_steps
        self._max_action = 0.1
        self._max_x = 2

        self.screen = None
        self.screen_width = 500
        self.screen_height = 500
        self.zoom = 50.
        self.random_start = random_start

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-self._max_x, high=self._max_x, shape=(2, ), dtype=np.float32
        )

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1., high=1., shape=(2, ), dtype=np.float32)

    def reset(self):
        self._point = np.sign(
            np.random.uniform(
                low=-self._max_x, high=self._max_x, size=self._point.shape
            )
        )
        self._step = 0
        return np.copy(self._point)

    def step(self, action):
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a *= self._action_scale
        a = np.clip(a, -self._max_action, self._max_action)

        self._point = np.clip(self._point + a, -self._max_x, self._max_x)

        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self._max_action)

        # dense reward
        reward = -dist / (self._max_x * np.sqrt(self._max_x))

        # completion bonus
        if done:
            reward += self._completion_bonus

        # steps count
        self._step += 1
        if self._step >= self._max_steps:
            done = True

        return np.copy(self._point), reward, done, {}

    def _to_screen(self, position):
        position = np.nan_to_num(position)
        return (
            int(self.screen_width / 2 + position[0] * self.zoom),
            int(self.screen_height / 2 - position[1] * self.zoom)
        )

    def render(self, mode="human"):
        if not PYGAME_ENABLED:
            return

        if self.screen is None and mode == "human":
            pygame.init()
            caption = "Point Environment"
            pygame.display.set_caption(caption)
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

        self.screen.fill((255, 255, 255))

        # draw grid
        for x in range(25):
            dx = -6. + x * 0.5
            pygame.draw.line(
                self.screen, DARK_COLOR if x % 2 == 0 else BRIGHT_COLOR,
                self._to_screen((dx, -10)), self._to_screen((dx, 10))
            )
        for y in range(25):
            dy = -6. + y * 0.5
            pygame.draw.line(
                self.screen, DARK_COLOR if y % 2 == 0 else BRIGHT_COLOR,
                self._to_screen((-10, dy)), self._to_screen((10, dy))
            )

        # draw goal
        pygame.draw.circle(
            self.screen, (255, 40, 0), self._to_screen(self._goal), 10, 0
        )

        # draw point
        pygame.draw.circle(
            self.screen, (40, 180, 10), self._to_screen(self._point), 10, 0
        )

        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            pass

    def close(self):
        if self.screen:
            pygame.quit()


register(
    id="PointEnv-v0",
    entry_point="_tests_rl_gym.point_env:PointEnv",
)
