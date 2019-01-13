import numpy as np
import torch
import torch.nn as nn


def get_out_features(head_net):
    head_modules = list(head_net.net.modules())
    last_linear = head_modules[-1]
    for m in head_modules[::-1]:
        if isinstance(last_linear, nn.Linear):
            break
        last_linear = m
    out_features = last_linear.out_features
    return out_features


def log1p_exp(input_tensor):
    """ Computationally stable function for computing log(1+exp(x)).
    """
    x = input_tensor * input_tensor.ge(0).to(torch.float32)
    res = x + torch.log1p(torch.exp(-torch.abs(input_tensor)))
    return res


def normal_sample(mu, sigma):
    """ Sample from multivariate Gaussian distribution z ~ N(z|mu,sigma)
    while supporting backpropagation through its mean and variance.
    """
    return mu + sigma * torch.randn_like(sigma)


def normal_log_prob(mu, sigma, z):
    """ Probability density function of multivariate Gaussian distribution
    N(z|mu,sigma).
    """
    normalization_constant = (-sigma.log() - 0.5 * np.log(2 * np.pi))
    square_term = -0.5 * ((z - mu) / sigma)**2
    log_prob_vec = normalization_constant + square_term
    log_prob = log_prob_vec.sum(1)
    return log_prob
