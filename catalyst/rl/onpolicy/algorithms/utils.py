import numpy as np


def create_gamma_matrix(tau, matrix_size):
    """
    Matrix of the following form
    --------------------
    1     y   y^2    y^3
    0     1     y    y^2
    0     0     1      y
    0     0     0      1
    --------------------
    for fast gae calculation
    """
    i = np.arange(matrix_size)
    j = np.arange(matrix_size)
    pow_ = i[None, :] - j[:, None]
    mat = np.power(tau, pow_) * (pow_ >= 0)
    return mat
