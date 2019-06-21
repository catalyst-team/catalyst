import numpy as np


def _structed2dict(array):
    if array.dtype.fields is not None:
        array = {
            key: array[key]
            for key in array.dtype.fields.keys()
        }
    return array


def _dict2structed(array):
    if isinstance(array, dict):
        capacity = 0
        dtype = []
        for key, value in array.items():
            capacity = len(value)
            dtype.append((key, value.dtype, value.shape[1:]))
        dtype = np.dtype(dtype)
        array_ = np.empty(capacity, dtype=dtype)
        for key, value in array.items():
            array_[key] = value
        array = array_
    return array


def preprocess_db_trajectory(trajectory):
    observations, actions, rewards, dones = trajectory
    observations = _structed2dict(observations)
    actions = _structed2dict(actions)
    trajectory = observations, actions, rewards, dones
    return trajectory


def postprocdess_db_trajectory(trajectory):
    observations, actions, rewards, dones = trajectory
    observations = _dict2structed(observations)
    actions = _dict2structed(actions)
    trajectory = observations, actions, rewards, dones
    return trajectory
