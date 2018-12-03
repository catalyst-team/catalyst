import numpy as np
import torch
import torch.nn as nn

# @TODO resolve issue with elu nonlinearity
ACTIVATIONS = {
    None: "sigmoid",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.ELU: "relu",
}


def log1p_exp(input_tensor):
    """ Computationally stable function for computing log(1+exp(x)).
    """
    x = input_tensor * input_tensor.ge(0).to(torch.float32)
    res = x + torch.log1p(
        torch.exp(-torch.abs(input_tensor)))
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
    normalization_constant = (
            - sigma.log()
            - 0.5 * np.log(2 * np.pi))
    square_term = -0.5 * ((z - mu) / sigma) ** 2
    log_prob_vec = normalization_constant + square_term
    log_prob = log_prob_vec.sum(1)
    return log_prob


def create_optimal_inner_init(nonlinearity, **kwargs):
    """ Create initializer for inner layers of policy and value networks
    based on their activation function (nonlinearity).
    """

    nonlinearity = ACTIVATIONS.get(nonlinearity, nonlinearity)
    assert isinstance(nonlinearity, str)

    if nonlinearity in ["sigmoid", "tanh"]:
        weignt_init_fn = nn.init.xavier_uniform_
        init_args = kwargs
    elif nonlinearity in ["relu", "leaky_relu"]:
        weignt_init_fn = nn.init.kaiming_normal_
        init_args = {**{"nonlinearity": nonlinearity}, **kwargs}
    else:
        raise NotImplemented

    def inner_init(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weignt_init_fn(layer.weight.data, **init_args)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)

    return inner_init


def out_init(layer):
    """ Initialization for output layers of policy and value networks typically
    used in deep reinforcement learning literature.
    """
    if isinstance(layer, nn.Linear):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)
