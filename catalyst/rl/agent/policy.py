import torch
import torch.nn as nn
from catalyst.contrib.registry import MODULES
from catalyst.contrib.modules import SquashingLayer, CouplingLayer
from catalyst.utils import normal_sample, normal_logprob

# log_sigma of Gaussian policy are capped at (LOG_SIG_MIN, LOG_SIG_MAX)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


def _distribution_forward(dist, action, logprob):
    bool_logprob = isinstance(logprob, bool) and logprob
    value_logprob = isinstance(logprob, torch.Tensor)

    if bool_logprob:
        # we need to compute logprob for current action
        action_logprob = dist.log_prob(action)
        return action, action_logprob
    elif value_logprob:
        # we need to compute logprob for external action
        action_logprob = dist.log_prob(logprob)
        return action, action_logprob
    else:
        # we need to compute current action only
        return action


class CategoricalPolicy(nn.Module):
    def forward(self, logits, logprob=None, deterministic=False):
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=1) \
            if deterministic \
            else dist.sample()

        return _distribution_forward(dist, action, logprob)


class BernoulliPolicy(nn.Module):
    def forward(self, logits, logprob=None, deterministic=False):
        dist = torch.distributions.Bernoulli(logits=logits)
        action = torch.gt(dist.probs, 0.5).float() \
            if deterministic \
            else dist.sample()

        return _distribution_forward(dist, action, logprob)


class DiagonalGaussPolicy(nn.Module):
    def forward(self, logits, logprob=None, deterministic=False):
        action_size = logits.shape[1] // 2
        loc, log_scale = logits[:, :action_size], logits[:, action_size:]
        log_scale = torch.clamp(log_scale, LOG_SIG_MIN, LOG_SIG_MAX)
        scale = torch.exp(log_scale)

        dist = torch.distributions.Normal(loc, scale)
        dist = torch.distributions.Independent(dist, 1)
        action = dist.mean if deterministic else dist.sample()

        return _distribution_forward(dist, action, logprob)


class SquashingGaussPolicy(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        super().__init__()
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, logits, logprob=None, deterministic=False):
        action_size = logits.shape[1] // 2
        loc, log_scale = logits[:, :action_size], logits[:, action_size:]
        log_scale = torch.clamp(log_scale, LOG_SIG_MIN, LOG_SIG_MAX)
        scale = torch.exp(log_scale)
        action = loc if deterministic else normal_sample(loc, scale)

        bool_logprob = isinstance(logprob, bool) and logprob
        value_logprob = isinstance(logprob, torch.Tensor)
        assert not value_logprob, "Not implemented behaviour"

        action_logprob = normal_logprob(loc, scale, action)
        action, action_logprob = \
            self.squashing_layer.forward(action, action_logprob)

        if bool_logprob:
            return action, action_logprob
        else:
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

    def forward(self, logits, logprob=None, deterministic=False):
        state_embedding = logits
        loc = torch.zeros((state_embedding.shape[0], self.action_size)).to(
            state_embedding.device
        )
        scale = torch.ones_like(loc).to(loc.device)
        action = loc if deterministic else normal_sample(loc, scale)

        bool_logprob = isinstance(logprob, bool) and logprob
        value_logprob = isinstance(logprob, torch.Tensor)
        assert not value_logprob, "Not implemented behaviour"

        action_logprob = normal_logprob(loc, scale, action)
        action, action_logprob = \
            self.coupling1.forward(action, state_embedding, action_logprob)
        action, action_logprob = \
            self.coupling2.forward(action, state_embedding, action_logprob)
        action, action_logprob = \
            self.squashing_layer.forward(action, action_logprob)

        if bool_logprob:
            return action, action_logprob
        else:
            return action


__all__ = ["CategoricalPolicy", "SquashingGaussPolicy", "RealNVPPolicy"]
