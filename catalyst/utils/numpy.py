import numpy as np


def get_one_hot(label: int, num_classes: int, smoothing: float = None) -> np.ndarray:
    """
    Applies OneHot vectorization to a giving scalar, optional with
    label smoothing as described in `Bag of Tricks for Image Classification
    with Convolutional Neural Networks`_.

    Args:
        label: scalar value to be vectorized
        num_classes: total number of classes
        smoothing (float, optional): if specified applies label smoothing
            from ``Bag of Tricks for Image Classification
            with Convolutional Neural Networks`` paper

    Returns:
        np.ndarray: a one-hot vector with shape ``(num_classes)``

    .. _Bag of Tricks for Image Classification with
        Convolutional Neural Networks: https://arxiv.org/abs/1812.01187
    """
    assert (
        num_classes is not None and num_classes > 0
    ), f"Expect num_classes to be > 0, got {num_classes}"

    assert (
        label is not None and 0 <= label < num_classes
    ), f"Expect label to be in [0; {num_classes}), got {label}"

    if smoothing is not None:
        assert (
            0.0 < smoothing < 1.0
        ), f"If smoothing is specified it must be in (0; 1), got {smoothing}"

        smoothed = smoothing / float(num_classes - 1)
        result = np.full((num_classes,), smoothed, dtype=np.float32)
        result[label] = 1.0 - smoothing

        return result

    result = np.zeros(num_classes, dtype=np.float32)
    result[label] = 1.0

    return result


__all__ = ["get_one_hot"]
