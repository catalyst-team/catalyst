import torch
import torch.nn as nn
from catalyst.rl.registry import MODULES
from .layers import SquashingLayer, CouplingLayer
from .utils import normal_sample, normal_log_prob


# log_sigma of Gaussian policy are capped at (LOG_SIG_MIN, LOG_SIG_MAX)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


class CategoricalPolicy(nn.Module):

    def forward(self, inputs, logprob=False, deterministic=False):
        dist = torch.distributions.Categorical(logits=inputs)
        action = torch.argmax(inputs, dim=1) \
            if deterministic \
            else dist.sample()
        flag_bool = isinstance(logprob, bool) and logprob
        flag_value = \
            not isinstance(logprob, bool) and logprob is not None
        if flag_bool or flag_value:
            # @TODO: refactor
            log_pi = dist.log_prob(logprob)
            return action, log_pi
        return action


class GaussPolicy(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        super().__init__()
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, inputs, logprob=False, deterministic=False):
        action_size = inputs.shape[1] // 2
        mu, log_sigma = inputs[:, :action_size], inputs[:, action_size:]
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(log_sigma)
        z = mu if deterministic else normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        action, log_pi = self.squashing_layer.forward(z, log_pi)

        if logprob:
            return action, log_pi
        return action


class RealNVPPolicy(nn.Module):
    def __init__(
        self,
        action_size,
        layer_fn,
        activation_fn=nn.ReLU,
        squashing_fn=nn.Tanh,
        bias=False
    ):
        super().__init__()
        activation_fn = MODULES.get_if_str(activation_fn)
        self.action_size = action_size

        self.coupling1 = CouplingLayer(
            action_size=action_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            bias=bias,
            parity="odd"
        )
        self.coupling2 = CouplingLayer(
            action_size=action_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            bias=bias,
            parity="even"
        )
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, inputs, logprob=False, deterministic=False):
        state_embedding = inputs
        mu = torch.zeros((state_embedding.shape[0], self.action_size)).to(
            state_embedding.device
        )
        sigma = torch.ones_like(mu).to(mu.device)
        z = mu if deterministic else normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        z, log_pi = self.coupling1.forward(z, state_embedding, log_pi)
        z, log_pi = self.coupling2.forward(z, state_embedding, log_pi)
        action, log_pi = self.squashing_layer.forward(z, log_pi)

        if logprob:
            return action, log_pi
        return action
