import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.utils import log1p_exp, get_out_features, \
    normal_sample, normal_log_prob

# log_sigma of Gaussian policy are capped at (LOG_SIG_MIN, LOG_SIG_MAX)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


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


class StateNet(nn.Module):
    def __init__(self, observation_net, head_net, policy_net, memory_net=None):
        super().__init__()
        self.observation_net = observation_net
        self.memory_net = memory_net
        self.head_net = head_net
        self.policy_net = policy_net

        self.out_features = get_out_features(head_net)

        self._forward_fn = None
        if isinstance(memory_net, LamaPooling):
            self._forward_fn = self._forward_lama
        elif isinstance(memory_net, nn.LSTM):
            raise NotImplementedError
            self._forward_fn = self._forward_rnn
        else:
            self._forward_fn = self._forward_ff

        self._policy_fn = None
        if isinstance(policy_net, GaussPolicy):
            self._policy_fn = self.policy_net.forward
        elif isinstance(policy_net, RealNVPPolicy):
            self._policy_fn = self.policy_net.forward
        else:
            self._policy_fn = lambda *args: args[0]

    def _forward_ff(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.observation_net(x)
        x = self.head_net(x)
        return x

    def _forward_rnn(self, observation, hidden_state):
        # @TODO
        raise NotImplementedError
        action, hidden_state = None, None
        return action, hidden_state

    def _forward_lama(self, observation):
        x = observation
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        batch_size, history_len, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.observation_net(x)

        x = x.view(batch_size, history_len, -1)
        x = self.memory_net(x)
        x = self.head_net(x)
        return x

    def forward(self, *args, with_log_pi=False, **kwargs):
        x = self._forward_fn(*args, **kwargs)
        x = self._policy_fn(x, with_log_pi)
        return x


class StateActionNet(nn.Module):
    def __init__(
        self, observation_net, action_net, bone_net, head_net, memory_net=None
    ):
        super().__init__()
        self.observation_net = observation_net
        self.action_net = action_net
        self.memory_net = memory_net
        self.bone_net = bone_net
        self.head_net = head_net

        self.out_features = get_out_features(head_net)

        self._forward_fn = None
        if isinstance(memory_net, LamaPooling):
            self._forward_fn = self._forward_lama
        elif isinstance(memory_net, nn.LSTM):
            raise NotImplementedError
            self._forward_fn = self._forward_rnn
        else:
            self._forward_fn = self._forward_ff

    def _forward_ff(self, observation, action):
        obs_ = observation.view(observation.shape[0], -1)
        if self.observation_net is not None:
            obs_ = self.observation_net(obs_)

        act_ = action.view(action.shape[0], -1)
        if self.action_net is not None:
            act_ = self.action_net(act_)

        x = torch.cat((obs_, act_), dim=1)
        x = self.bone_net(x)
        x = self.head_net(x)
        return x

    def _forward_rnn(self, observation, action, hidden_state):
        # @TODO
        raise NotImplementedError
        action, hidden_state = None, None
        return action, hidden_state

    def _forward_lama(self, observation, action):
        x = observation
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        batch_size, history_len, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.observation_net(x)

        x = x.view(batch_size, history_len, -1)
        observation_ = self.memory_net(x)

        # @TODO: add option to collapse observations based on action
        x = action.view(action.shape[0], -1)
        action_ = self.action_net(x)

        x = torch.cat((observation_, action_), dim=1)
        x = self.head_net(x)
        return x

    def forward(self, *args, **kwargs):
        value = self._forward_fn(*args, **kwargs)
        return value


class SquashingLayer(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        """ Layer that squashes samples from some distribution to be bounded.
        """
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry
        self.squashing_fn = Registry.name2nn(squashing_fn)()

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


class GaussPolicy(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        super().__init__()
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, inputs, with_log_pi=True):
        action_size = inputs.shape[1] // 2
        mu, log_sigma = inputs[:, :action_size], inputs[:, action_size:]
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(log_sigma)
        z = normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        action, log_pi = self.squashing_layer.forward(z, log_pi)

        if with_log_pi:
            return action, log_pi
        return action


class RealNVPPolicy(nn.Module):
    def __init__(
        self,
        action_size,
        layer_fn,
        activation_fn=nn.ReLU,
        squashing_fn=nn.Tanh,
        norm_fn=None,
        bias=False
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry
        activation_fn = Registry.name2nn(activation_fn)
        self.action_size = action_size

        self.coupling1 = CouplingLayer(
            action_size=action_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias,
            parity="odd"
        )
        self.coupling2 = CouplingLayer(
            action_size=action_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias,
            parity="even"
        )
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, inputs, with_log_pi=True):
        state_embedding = inputs
        mu = torch.zeros((state_embedding.shape[0], self.action_size)).to(
            state_embedding.device
        )
        sigma = torch.ones_like(mu).to(mu.device)
        z = normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        z, log_pi = self.coupling1.forward(z, state_embedding, log_pi)
        z, log_pi = self.coupling2.forward(z, state_embedding, log_pi)
        action, log_pi = self.squashing_layer.forward(z, log_pi)

        if with_log_pi:
            return action, log_pi
        return action


class CouplingLayer(nn.Module):
    def __init__(
        self,
        action_size,
        layer_fn,
        activation_fn=nn.ReLU,
        norm_fn=None,
        bias=True,
        parity="odd"
    ):
        """ Conditional affine coupling layer used in Real NVP Bijector.
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
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry

        layer_fn = Registry.name2nn(layer_fn)
        activation_fn = Registry.name2nn(activation_fn)
        norm_fn = Registry.name2nn(norm_fn)

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
