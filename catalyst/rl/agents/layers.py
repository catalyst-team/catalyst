import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.utils import log1p_exp
from catalyst.rl.registry import MODULES


class TemporalAttentionPooling(nn.Module):
    name2fn = {
        "softmax": nn.Softmax(dim=1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }

    def __init__(self, features_in, pooling=None):
        super().__init__()
        self.features_in = features_in
        pooling = pooling or "softmax"

        self.attention_pooling = nn.Sequential(
            nn.Conv1d(
                in_channels=features_in,
                out_channels=1,
                kernel_size=1,
                bias=True
            ), TemporalAttentionPooling.name2fn[pooling]
        )
        self.attention_pooling.apply(outer_init)

    def forward(self, features):
        """
        :param features: [batch_size, history_len, feature_size]
        :return:
        """
        x = features
        batch_size, history_len, feature_size = x.shape

        x = x.view(batch_size, history_len, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attention_pooling(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)

        return x_attn


class LamaPooling(nn.Module):
    available_poolings = [
        "last", "avg_all", "avg", "max_all", "max", "softmax_all", "softmax",
        "tanh_all", "tanh", "sigmoid_all", "sigmoid"
    ]

    def __init__(self, features_in, poolings=None):
        super().__init__()
        self.features_in = features_in
        self.poolings = poolings or ["last", "avg", "max", "softmax"]
        self.features_out = features_in * len(self.poolings)

        self.poolings = nn.ModuleDict({
            k: self._prepare_for_pooling(k, self.features_in)
            for k in self.poolings
        })

    @staticmethod
    def _prepare_for_pooling(key, features_in):
        if any([x in key for x in ["softmax", "tanh", "sigmoid"]]):
            key = key.split("_", 1)[0]
            pooling = TemporalAttentionPooling(
                features_in=features_in, pooling=key
            )
            return pooling

    def _pooling_fn(self, key, features):
        x = features
        key_ = key.split("_", 1)

        if key != "last" and len(key_) == 1:
            # all except last
            x = x[:, :-1, :]
            key = key_[0]

        if key == "last":
            x_out = x[:, -1:, :]
        elif key == "avg":
            x_out = x.mean(1, keepdim=True)
        elif key == "max":
            x_out = x.max(1, keepdim=True)[0]
        elif any([x in key for x in ["softmax", "tanh", "sigmoid"]]):
            x_out = self.poolings[key](x)
        else:
            raise NotImplementedError

        return x_out

    def forward(self, features):
        """
        :param features: [batch_size, history_len, feature_size]
        :return:
        """
        x = features
        batch_size, history_len, feature_size = x.shape

        features_ = []
        for key in self.poolings:
            pooling = self._pooling_fn(key, features)
            features_.append(pooling)
        x = torch.cat(features_, dim=1)
        x = x.view(batch_size, -1)

        return x


class SquashingLayer(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        """
        Layer that squashes samples from some distribution to be bounded.
        """
        super().__init__()

        self.squashing_fn = MODULES.get_if_str(squashing_fn)()

    def forward(self, action, log_pi):
        # compute log det jacobian of squashing transformation
        if isinstance(self.squashing_fn, nn.Tanh):
            log2 = torch.log(torch.tensor(2.0).to(action.device))
            log_det_jacobian = 2 * (log2 + action - log1p_exp(2 * action))
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif isinstance(self.squashing_fn, nn.Sigmoid):
            log_det_jacobian = -action - 2 * log1p_exp(-action)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif isinstance(self.squashing_fn, None):
            return action, log_pi
        else:
            raise NotImplementedError
        action = self.squashing_fn.forward(action)
        log_pi = log_pi - log_det_jacobian
        return action, log_pi


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

        layer_fn = MODULES.get_if_str(layer_fn)
        activation_fn = MODULES.get_if_str(activation_fn)

        self.parity = parity
        if self.parity == "odd":
            self.copy_size = action_size // 2
        else:
            self.copy_size = action_size - action_size // 2

        self.scale_prenet = SequentialNet(
            hiddens=[action_size * 2 + self.copy_size, action_size],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias
        )
        self.scale_net = SequentialNet(
            hiddens=[action_size, action_size - self.copy_size],
            layer_fn=layer_fn,
            activation_fn=None,
            norm_fn=None,
            bias=True
        )

        self.translation_prenet = SequentialNet(
            hiddens=[action_size * 2 + self.copy_size, action_size],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias
        )
        self.translation_net = SequentialNet(
            hiddens=[action_size, action_size - self.copy_size],
            layer_fn=layer_fn,
            activation_fn=None,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.scale_prenet.apply(inner_init)
        self.scale_net.apply(outer_init)
        self.translation_prenet.apply(inner_init)
        self.translation_net.apply(outer_init)

    def forward(self, action, state_embedding, log_pi):
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
        log_pi = log_pi - log_det_jacobian

        return action, log_pi
