from typing import Dict

import numpy as np
from scipy.signal import lfilter


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)


def geometric_cumsum(alpha, x):
    """
    Calculate future accumulated sums for each element
    in a list with an exponential factor.

    Given input data :math:`x_1, \dots, x_n`  # noqa: E501, W605
    and exponential factor :math:`\alpha\in [0, 1]`,  # noqa: E501, W605
    it returns an array :math:`y`
    with the same length and each element is calculated as following

    .. math::
        y_i = x_i + \alpha x_{i+1} + \alpha^2 x_{i+2} + \dots + \alpha^{n-i-1}x_{n-1} + \alpha^{n-i}x_{n}  # noqa: E501, W605

    .. note::
        To gain the optimal runtime speed, we use ``scipy.signal.lfilter``

    Example:
        >>> geometric_cumsum(0.1, [[1, 1], [2, 2], [3, 3], [4, 4]])
        array([[1.234, 1.234], [2.34 , 2.34 ], [3.4  , 3.4  ], [4.   , 4.   ]])

    Args:
        alpha (float): exponential factor between zero and one.
        x (np.ndarray): input data, [trajectory_len, num_atoms]

    Returns:
        out (np.ndarray): calculated data

    source: https://github.com/zuoxingdong/lagom

    """
    x = np.asarray(x)
    assert x.ndim == 2
    return lfilter([1], [1, -alpha], x[::-1, :], axis=0)[::-1, :]


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


def get_one_hot(
    label: int, num_classes: int, smoothing: float = None
) -> np.ndarray:
    """
    Applies OneHot vectorization to a giving scalar, optional with
    label smoothing from https://arxiv.org/abs/1812.01187

    Args:
        label (int): scalar value to be vectorized
        num_classes (int): total number of classes
        smoothing (float, optional): if specified applies label smoothing
            from ``Bag of Tricks for Image Classification
            with Convolutional Neural Networks`` paper

    Returns:
        np.ndarray: a one-hot vector with shape ``(num_classes,)``
    """
    assert num_classes is not None and num_classes > 0, \
        f"Expect num_classes to be > 0, got {num_classes}"

    assert label is not None and 0 <= label < num_classes, \
        f"Expect label to be in [0; {num_classes}), got {label}"

    if smoothing is not None:
        assert 0.0 < smoothing < 1.0, \
            f"If smoothing is specified it must be in (0; 1), got {smoothing}"

        smoothed = smoothing / float(num_classes - 1)
        result = np.full((num_classes, ), smoothed, dtype=np.float32)
        result[label] = 1.0 - smoothing

        return result

    result = np.zeros(num_classes, dtype=np.float32)
    result[label] = 1.0

    return result
