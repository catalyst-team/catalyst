import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.utils import create_optimal_inner_init, outer_init, log1p_exp
from catalyst.contrib.registry import MODULES


class SquashingLayer(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        """
        Layer that squashes samples from some distribution to be bounded.
        """
        super().__init__()

        self.squashing_fn = MODULES.get_if_str(squashing_fn)()

    def forward(self, action, action_logprob):
        # compute log det jacobian of squashing transformation
        if isinstance(self.squashing_fn, nn.Tanh):
            log2 = torch.log(torch.tensor(2.0).to(action.device))
            log_det_jacobian = 2 * (log2 + action - log1p_exp(2 * action))
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif isinstance(self.squashing_fn, nn.Sigmoid):
            log_det_jacobian = -action - 2 * log1p_exp(-action)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif self.squashing_fn is None:
            return action, action_logprob
        else:
            raise NotImplementedError
        action = self.squashing_fn.forward(action)
        action_logprob = action_logprob - log_det_jacobian
        return action, action_logprob


class CouplingLayer(nn.Module):
    def __init__(
        self,
        action_size,
        layer_fn,
        activation_fn=nn.ReLU,
        bias=True,
        parity="odd"
    ):
        """
        Conditional affine coupling layer used in Real NVP Bijector.
        Original paper: https://arxiv.org/abs/1605.08803
        Adaptation to RL: https://arxiv.org/abs/1804.02808
        Important notes
        ---------------
        1. State embeddings are supposed to have size (action_size * 2).
        2. Scale and translation networks used in the Real NVP Bijector
        both have one hidden layer of (action_size) (activation_fn) units.
        3. Parity ("odd" or "even") determines which part of the input
        is being copied and which is being transformed.
        """
        super().__init__()
        assert parity in ["odd", "even"]

        layer_fn = MODULES.get_if_str(layer_fn)

        self.parity = parity
        if self.parity == "odd":
            self.copy_size = action_size // 2
        else:
            self.copy_size = action_size - action_size // 2

        self.scale_prenet = SequentialNet(
            hiddens=[action_size * 2 + self.copy_size, action_size],
            layer_fn={"module": layer_fn, "bias": bias},
            activation_fn=activation_fn,
            norm_fn=None,
        )
        self.scale_net = SequentialNet(
            hiddens=[action_size, action_size - self.copy_size],
            layer_fn={"module": layer_fn, "bias": True},
            activation_fn=None,
            norm_fn=None,
        )

        self.translation_prenet = SequentialNet(
            hiddens=[action_size * 2 + self.copy_size, action_size],
            layer_fn={"module": layer_fn, "bias": bias},
            activation_fn=activation_fn,
            norm_fn=None,
        )
        self.translation_net = SequentialNet(
            hiddens=[action_size, action_size - self.copy_size],
            layer_fn={"module": layer_fn, "bias": True},
            activation_fn=None,
            norm_fn=None,
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.scale_prenet.apply(inner_init)
        self.scale_net.apply(outer_init)
        self.translation_prenet.apply(inner_init)
        self.translation_net.apply(outer_init)

    def forward(self, action, state_embedding, action_logprob):
        if self.parity == "odd":
            action_copy = action[:, :self.copy_size]
            action_transform = action[:, self.copy_size:]
        else:
            action_copy = action[:, -self.copy_size:]
            action_transform = action[:, :-self.copy_size]

        x = torch.cat((state_embedding, action_copy), dim=1)

        t = self.translation_prenet(x)
        t = self.translation_net(t)

        s = self.scale_prenet(x)
        s = self.scale_net(s)

        out_transform = t + action_transform * torch.exp(s)

        if self.parity == "odd":
            action = torch.cat((action_copy, out_transform), dim=1)
        else:
            action = torch.cat((out_transform, action_copy), dim=1)

        log_det_jacobian = s.sum(dim=1)
        action_logprob = action_logprob - log_det_jacobian

        return action, action_logprob


__all__ = ["SquashingLayer", "CouplingLayer"]
