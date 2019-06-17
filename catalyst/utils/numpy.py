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
