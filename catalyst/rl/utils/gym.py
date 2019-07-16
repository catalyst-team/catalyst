import numpy as np
from collections import OrderedDict
from gym import spaces


def extend_space(space, history_len) -> spaces.Space:
    def _extend_to_history_len(np_array):
        return np.concatenate(
            history_len * [np.expand_dims(np_array, 0)], axis=0
        )

    if isinstance(space, spaces.Discrete):
        result = spaces.MultiDiscrete([history_len, space.n])
    elif isinstance(space, spaces.MultiDiscrete):
        nvec = np.hstack(
            (history_len * np.ones((space.nvec.shape[0], 1)), space.nvec)
        )
        result = spaces.MultiDiscrete(nvec)
    elif isinstance(space, spaces.Box):
        result = spaces.Box(
            low=_extend_to_history_len(space.low),
            high=_extend_to_history_len(space.high),
            # shape=(history_len,) + space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, spaces.Tuple):
        result = []
        for value in space.spaces:
            result.append(extend_space(value, history_len))
        result = spaces.Tuple(result)
    elif isinstance(space, spaces.Dict):
        result = OrderedDict()
        for key, value in space.spaces.items():
            result[key] = extend_space(value, history_len)
        result = spaces.Dict(result)
    else:
        raise NotImplementedError("not yet implemented")

    return result


__all__ = ["extend_space"]
