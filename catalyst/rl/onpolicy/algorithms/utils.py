import numpy as np
from scipy.signal import lfilter


def geometric_cumsum(alpha, x):
    r"""
    Adapopted from https://github.com/zuoxingdong/lagom
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    assert x.ndim == 2
    return lfilter([1], [1, -alpha], x[:, ::-1], axis=1)[:, ::-1]
