import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.utils import log1p_exp, normal_sample, normal_log_prob

# log_sigma of Gaussian policy are capped at (LOG_SIG_MIN, LOG_SIG_MAX)
from catalyst.rl.registry import MODULES

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


class SquashingLayer(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        """ Layer that squashes samples from some distribution to be bounded.
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


class GaussPolicy(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        super().__init__()
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, inputs, with_log_pi=True, deterministic=False):
        action_size = inputs.shape[1] // 2
        mu, log_sigma = inputs[:, :action_size], inputs[:, action_size:]
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(log_sigma)
        z = mu if deterministic else normal_sample(mu, sigma)
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

    def forward(self, inputs, with_log_pi=True, deterministic=False):
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

        if with_log_pi:
            return action, log_pi
        return action


class CouplingLayer(nn.Module):
    def __init__(
        self,
        action_size,
        layer_fn,
        activation_fn=nn.ReLU,
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


class ValueHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_atoms: int = 1,
        bias: bool = False,
        distribution: str = None,
        values_range: tuple = None
    ):
        super().__init__()

        self.out_features = out_features
        self.num_atoms = num_atoms
        self.distribution = distribution
        self.values_range = values_range

        if distribution is None:  # mean case
            assert values_range is None and num_atoms == 1
        elif distribution == "categorical":
            assert values_range is not None and num_atoms > 1
        elif distribution == "quantile":
            assert values_range is None and num_atoms > 1
        else:
            raise NotImplementedError()

        self.net = nn.Linear(
            in_features=in_features,
            out_features=out_features * num_atoms,
            bias=bias
        )
        self.apply(outer_init)

    def forward(self, inputs):
        x: torch.Tensor = \
            self.net(inputs).view(-1, self.out_features, self.num_atoms)
        x = x.squeeze_(dim=-1)
        return x


class PolicyHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        policy_type: str = None,
        out_activation: nn.Module = None
    ):
        super().__init__()

        # @TODO: refactor
        layer_fn = nn.Linear
        activation_fn = nn.ReLU
        squashing_fn = nn.Tanh
        bias = True

        if policy_type == "gauss":
            head_size = out_features * 2
            policy_net = GaussPolicy(squashing_fn)
        elif policy_type == "real_nvp":
            head_size = out_features * 2
            policy_net = RealNVPPolicy(
                action_size=out_features,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                squashing_fn=squashing_fn,
                bias=bias
            )
        else:
            head_size = out_features
            policy_net = None
            policy_type = "logits"

        self.policy_type = policy_type

        head_net = SequentialNet(
            hiddens=[in_features, head_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )
        head_net.apply(outer_init)
        self.head_net = head_net

        self.policy_net = policy_net
        self._policy_fn = None
        if policy_net is None:
            self._policy_fn = lambda *args: args[0]
        elif isinstance(policy_net, (GaussPolicy, RealNVPPolicy)):
            self._policy_fn = policy_net.forward
        else:
            raise NotImplementedError

    def forward(self, inputs, with_log_pi=False, deterministic=False):
        x = self.head_net(inputs)
        x = self._policy_fn(x, with_log_pi, deterministic)
        return x


class StateNet(nn.Module):
    """
    Abstract network, that takes some tensor T of shape [bs; history_len; ...]
    and outputs some representation tensor R of shape [bs; representation_size]

    input_T [bs; history_len; in_features]
        -> observation_net (aka observation_encoder) ->
    observations_representations [bs; history_len; obs_features]
        -> aggregation_net (flatten in simplified case)->
    aggregated_representation [bs; hid_features]
        -> main_net ->
    output_T [bs; representation_size]
    """

    def __init__(
        self,
        main_net: nn.Module,
        observation_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        super().__init__()
        self.main_net = main_net
        self.observation_net = observation_net or (lambda x: x)
        self.aggregation_net = aggregation_net

        self._forward_fn = None
        if aggregation_net is None:
            self._forward_fn = self._forward_ff
        elif isinstance(aggregation_net, LamaPooling):
            self._forward_fn = self._forward_lama
        else:
            raise NotImplementedError

    def _forward_ff(self, state):
        x = state.view(state.shape[0], -1)
        x = self.observation_net(x)
        x = self.main_net(x)
        return x

    def _forward_lama(self, state):
        x = state
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        if isinstance(self.observation_net, nn.Module):
            batch_size, history_len, feature_size = x.shape
            x = x.view(-1, feature_size)
            x = self.observation_net(x)
            x = x.view(batch_size, history_len, -1)

        x = self.aggregation_net(x)

        x = self.main_net(x)
        return x

    def forward(self, state):
        x = self._forward_fn(state)
        return x

    @classmethod
    def get_from_params(
        cls,
        observation_net_params=None,
        aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        assert observation_net_params is not None
        assert aggregation_net_params is None, "Lama is not implemented yet"

        observation_net = SequentialNet(**observation_net_params)
        main_net = SequentialNet(**main_net_params)
        net = cls(main_net=main_net, observation_net=observation_net)
        return net


class StateActionNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        observation_net: nn.Module = None,
        action_net: nn.Module = None,
        aggregation_net: nn.Module = None
    ):
        super().__init__()
        self.main_net = main_net
        self.observation_net = observation_net or (lambda x: x)
        self.action_net = action_net or (lambda x: x)
        self.aggregation_net = aggregation_net

        self._forward_fn = None
        if aggregation_net is None:
            self._forward_fn = self._forward_ff
        elif isinstance(aggregation_net, LamaPooling):
            self._forward_fn = self._forward_lama
        else:
            raise NotImplementedError

    def _forward_ff(self, state, action):
        state_ = state.view(state.shape[0], -1)
        state_ = self.observation_net(state_)

        action_ = action.view(action.shape[0], -1)
        action_ = self.action_net(action_)

        x = torch.cat((state_, action_), dim=1)
        x = self.main_net(x)
        return x

    def _forward_lama(self, state, action):
        state_ = state
        if len(state_.shape) < 3:
            state_ = state_.unsqueeze(1)

        if isinstance(self.observation_net, nn.Module):
            batch_size, history_len, feature_size = state_.shape
            state_ = state_.view(-1, feature_size)
            state_ = self.observation_net(state_)
            state_ = state_.view(batch_size, history_len, -1)

        state_ = self.aggregation_net(state_)

        # @TODO: add option to collapse observations based on action
        action_ = action.view(action.shape[0], -1)
        action_ = self.action_net(action_)

        x = torch.cat((state_, action_), dim=1)
        x = self.main_net(x)
        return x

    def forward(self, state, action):
        x = self._forward_fn(state, action)
        return x

    @classmethod
    def get_from_params(
        cls,
        observation_net_params=None,
        action_net_params=None,
        aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        assert observation_net_params is not None
        assert action_net_params is not None
        assert aggregation_net_params is None, "Lama is not implemented yet"

        observation_net = SequentialNet(**observation_net_params)
        action_net = SequentialNet(**action_net_params)
        main_net = SequentialNet(**main_net_params)
        net = cls(
            observation_net=observation_net,
            action_net=action_net,
            main_net=main_net
        )
        return net
