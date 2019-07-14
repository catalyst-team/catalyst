import numpy as np


def hyperbolic_gammas(gamma_max, k, num_heads):
    # Formula (27) from https://arxiv.org/pdf/1902.06865.pdf
    b = np.exp(np.log(1 - gamma_max**(1 / k)) / num_heads)
    gammas = (1 - b**(np.arange(num_heads) + 1))**k
    return gammas
