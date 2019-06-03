import numpy as np
from scipy.signal import lfilter


def geometric_cumsum(alpha, x):
    """
    Adapted from https://github.com/zuoxingdong/lagom
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    assert x.ndim == 2
    return lfilter([1], [1, -alpha], x[:, ::-1], axis=1)[:, ::-1]


def append_dict(dict1, dict2):
    """
    Appends dict2 with the same keys as dict1 to dict1
    """
    for key in dict1.keys():
        dict1[key] = np.concatenate((dict1[key], dict2[key]))
    return dict1
