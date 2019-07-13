from gym.core import Wrapper
from catalyst.rl.utils.buffer import get_buffer
from catalyst.utils.dynamic_array import DynamicArray
import numpy as np


class RawTrajectoryWrapper(Wrapper):
    def __init__(
        self,
        env,
        allow_early_resets=False,
        initial_capacity=1e3,
    ):
        """
        Wrapper which saves the raw trajectories of the environment
        into the info dict, which are then pushed to the database.
        """
        super().__init__(env)
        self.allow_early_resets = allow_early_resets
        self.needs_reset = True
        self.num_steps = 0
        self.initial_capacity = initial_capacity
        self._init_buffers()

    def _init_buffers(self):
        sample_size = 3
        observations_, observations_dtype = get_buffer(
            capacity=sample_size,
            space=self.env.observation_space,
            mode="numpy"
        )
        observations_shape = (None,) \
            if observations_.dtype.fields is not None \
            else (None,) + tuple(self.env.observation_space.shape)
        self.observations = DynamicArray(
            array_or_shape=observations_shape,
            capacity=int(self.initial_capacity),
            dtype=observations_dtype
        )

        actions_, actions_dtype = get_buffer(
            capacity=sample_size, space=self.env.action_space, mode="numpy"
        )
        actions_shape = (None,) \
            if actions_.dtype.fields is not None \
            else (None,) + tuple(self.env.action_space.shape)
        self.actions = DynamicArray(
            array_or_shape=actions_shape,
            capacity=int(self.initial_capacity),
            dtype=actions_dtype
        )

        self.rewards = DynamicArray(
            array_or_shape=(None, ),
            dtype=np.float32,
            capacity=int(self.initial_capacity)
        )
        self.dones = DynamicArray(
            array_or_shape=(None, ),
            dtype=np.bool,
            capacity=int(self.initial_capacity)
        )

    def _reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Wrap your env with "
                "RawTrajectoryWrapper(env, allow_early_resets=True)"
            )
        self.needs_reset = False

    def _init_with_observation(self, observation):
        self.observations.append(observation)

    def _put_transition(self, transition):
        """
        transition = [o_tp1, a_t, r_t, d_t]
        """
        o_tp1, a_t, r_t, d_t = transition
        self.observations.append(o_tp1)
        self.actions.append(a_t)
        self.rewards.append(r_t)
        self.dones.append(d_t)

    def _update(self, ob, rew, done, action, info):
        self._put_transition((ob, action, rew, done))
        assert isinstance(info, dict)
        if done:
            self.needs_reset = True
            info["raw_trajectory"] = (
                np.array(self.observations[:-1]), np.array(self.actions),
                np.array(self.rewards), np.array(self.dones)
            )
        return info

    def reset(self, **kwargs):
        self._reset_state()
        initial_state = self.env.reset(**kwargs)
        self._init_buffers()
        self._init_with_observation(initial_state)
        return initial_state

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        info = self._update(ob, rew, done, action, info)
        return (ob, rew, done, info)
