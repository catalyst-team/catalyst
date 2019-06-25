from typing import Dict

import numpy as np
from scipy.signal import lfilter


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)


def geometric_cumsum(alpha, x):
    """
    Adapted from https://github.com/zuoxingdong/lagom
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    assert x.ndim == 2
    return lfilter([1], [1, -alpha], x[:, ::-1], axis=1)[:, ::-1]


def structed2dict(array: np.ndarray):
    if isinstance(array, (np.ndarray, np.void)) \
            and array.dtype.fields is not None:
        array = {key: array[key] for key in array.dtype.fields.keys()}
    return array


def dict2structed(array: Dict):
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
