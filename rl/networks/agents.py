import torch
import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.networks.utils import normal_sample, normal_log_prob
from catalyst.rl.networks.misc_layers import SquashingLayer, CouplingLayer

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
            ),
            TemporalAttentionPooling.name2fn[pooling]
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
        x_attn = (self.attn(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)

        return x_attn


class LamaPooling(nn.Module):
    available_poolings = [
        "last",
        "avg_all", "avg",
        "max_all", "max",
        "softmax_all", "softmax",
        "tanh_all", "tanh",
        "sigmoid_all", "sigmoid"
    ]

    def __init__(self, features_in, poolings=None):
        super().__init__()
        self.features_in = features_in
        self.poolings = poolings or ["last", "avg", "max", "softmax"]
        self.features_out = features_in * len(poolings)

        for key in self.poolings:
            self._prepare_for_pooling(key)

    def _prepare_for_pooling(self, key):
        if any([x in key for x in ["softmax", "tanh", "sigmoid"]]):
            key = key.split("_", 1)[0]
            pooling = TemporalAttentionPooling(
                features_in=self.features_in,
                pooling=key)
            # @TODO: fix hack
            setattr(self, f"{key}_pooling", pooling)

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
            pooling = getattr(self, f"{key}_pooling")
            x_out = pooling(x)
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
        x = torch.cat(features_, dim=1, )
        x = x.view(batch_size, -1)

        return x


class Actor(nn.Module):
    """
    Actor which learns deterministic policy.
    """

    def __init__(self, observation_net, policy_net, memory_net=None):
        super().__init__()
        self.observation_net = observation_net
        self.memory_net = memory_net
        self.policy_net = policy_net
        self._forward_fn = None

        if memory_net is not None and memory_net == "lama":
            self._forward_fn = self._forward_lama
        elif memory_net is not None and memory_net == "rnn":
            self._forward_fn = self._forward_rnn
        else:
            self._forward_fn = self._forward_ff

    def _forward_ff(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.observation_net(x)
        x = self.policy_net(x)
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
        x = self.policy_net(x)
        return x

    def forward(self, *args, **kwargs):
        action = self._forward_fn(*args, **kwargs)
        return action

    @staticmethod
    def create_actor(
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            activation_fn=nn.ReLU,
            dropout=None,
            norm_fn=None,
            bias=True,
            layer_order=None,
            residual=False,
            out_activation=None,
            memory_type=None,
            **kwargs):
        assert len(kwargs) == 0
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            state_size = reduce(lambda x, y: x * y, state_shape)

            observation_net = SequentialNet(
                hiddens=[state_size] + hiddens[:-1],
                layer_fn=layer_fn,
                dropout=dropout,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias,
                layer_order=layer_order,
                residual=residual
            )
        elif len(state_shape) in [3, 4]:
            # cnn case: one image or several one @TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        if memory_type == "lama":
            raise NotImplementedError
        elif memory_type == "rnn":
            raise NotImplementedError
        else:
            memory_net = None
            memory_out = hiddens[-1]

        policy_net = SequentialNet(
            hiddens=[memory_out, action_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        observation_net.apply(inner_init)
        policy_net.apply(outer_init)

        actor_net = Actor(
            observation_net=observation_net,
            memory_net=memory_net,
            policy_net=policy_net)

        return actor_net


class ActorDeprecated(nn.Module):
    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=nn.Tanh
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        state_size = reduce(lambda x, y: x * y, state_shape)

        self.feature_net = SequentialNet(
            hiddens=[state_size] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.policy_net = SequentialNet(
            hiddens=[hiddens[-1], action_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.policy_net.apply(outer_init)

    def forward(self, states):
        x = states.view(states.shape[0], -1)
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)
        return x


class LamaActor(nn.Module):
    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=nn.Tanh
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        state_size = state_shape[-1]

        self.feature_net = SequentialNet(
            hiddens=[state_size] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.attn = nn.Sequential(
            nn.Conv1d(
                in_channels=hiddens[-1],
                out_channels=1,
                kernel_size=1,
                bias=True
            ), nn.Softmax(dim=1)
        )
        self.feature_net2 = SequentialNet(
            hiddens=[hiddens[-1] * 4, hiddens[-1]],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.policy_net = SequentialNet(
            hiddens=[hiddens[-1], action_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.attn.apply(outer_init)
        self.feature_net2.apply(inner_init)
        self.policy_net.apply(outer_init)

    def forward(self, states):
        if len(states.shape) < 3:
            states = states.unsqueeze(1)
        bs, ln, fd = states.shape
        x = states.view(-1, fd)
        x = self.feature_net(x)

        x = x.view(bs, ln, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attn(x_a) * x_a).transpose(1, 2)

        x_avg = x.mean(1, keepdim=True)
        x_max = x.max(1, keepdim=True)[0]
        x_attn = x_attn.mean(1, keepdim=True)
        x_last = x[:, -1:, :]
        x = torch.cat([x_last, x_avg, x_max, x_attn], dim=1)
        x = x.view(bs, -1)

        x = self.feature_net2(x)
        x = self.policy_net(x)
        return x


class GaussActor(nn.Module):
    """ Actor which learns mean and standard deviation of Gaussian
    stochastic policy. Actions obtained from the policy are squashed
    with (out_activation).
    """

    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=nn.Sigmoid
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        self.n_action = action_size

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        state_size = reduce(lambda x, y: x * y, state_shape)

        self.feature_net = SequentialNet(
            hiddens=[state_size] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.policy_net = SequentialNet(
            hiddens=[hiddens[-1], action_size * 2],
            layer_fn=nn.Linear,
            activation_fn=None,
            norm_fn=None,
            bias=bias
        )
        self.squasher = SquashingLayer(out_activation)

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.policy_net.apply(outer_init)

    def forward(self, observation, with_log_pi=False):
        observation = observation.view(observation.shape[0], -1)
        x = observation
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)

        mu, log_sigma = x[:, :self.n_action], x[:, self.n_action:]
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(log_sigma)
        z = normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        action, log_pi = self.squasher.forward(z, log_pi)

        if with_log_pi:
            return action, log_pi, mu, log_sigma
        return action


class RealNVPActor(nn.Module):
    """ Actor which learns policy based on Real NVP Bijector.
    Such policy transforms samples from N(z|0,I) into actions and
    then squashes them with (out activation).
    """

    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=nn.Sigmoid
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        self.n_action = action_size

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        state_size = reduce(lambda x, y: x * y, state_shape)

        self.feature_net = SequentialNet(
            hiddens=[state_size] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.embedding_net = SequentialNet(
            hiddens=[hiddens[-1], action_size * 2],
            layer_fn=layer_fn,
            activation_fn=None,
            norm_fn=norm_fn,
            bias=bias
        )

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

        self.squasher = SquashingLayer(out_activation)

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.embedding_net.apply(inner_init)

    def forward(self, observation, with_log_pi=False):
        observation = observation.view(observation.shape[0], -1)
        x = observation
        x = self.feature_net.forward(x)
        state_embedding = self.embedding_net.forward(x)

        mu = torch.zeros((observation.shape[0], self.n_action)).to(x.device)
        sigma = torch.ones_like(mu).to(x.device)
        z = normal_sample(mu, sigma)
        log_pi = normal_log_prob(mu, sigma, z)
        z, log_pi = self.coupling1.forward(z, state_embedding, log_pi)
        z, log_pi = self.coupling2.forward(z, state_embedding, log_pi)
        action, log_pi = self.squasher.forward(z, log_pi)

        if with_log_pi:
            return action, log_pi
        return action


class CriticDeprecated(nn.Module):
    """ Critic which learns state-action value function Q(s,a).
    """

    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            concat_at=1,
            n_atoms=1,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=None
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn
        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        self.n_atoms = n_atoms

        state_size = reduce(lambda x, y: x * y, state_shape)

        if concat_at > 0:
            hiddens_ = [state_size] + hiddens[0:concat_at]
            self.observation_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )
            hiddens_ = \
                [hiddens[concat_at - 1] + action_size] + hiddens[concat_at:]
            self.feature_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )
        else:
            self.observation_net = None
            hiddens_ = [state_size + action_size] + hiddens
            self.feature_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )

        self.value_net = SequentialNet(
            hiddens=[hiddens[-1], n_atoms],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        if self.observation_net is not None:
            self.observation_net.apply(inner_init)
        self.feature_net.apply(inner_init)
        self.value_net.apply(outer_init)

    def forward(self, observation, action):
        observation = observation.view(observation.shape[0], -1)
        if self.observation_net is not None:
            observation = self.observation_net(observation)
        x = torch.cat((observation, action), dim=1)
        x = self.feature_net.forward(x)
        x = self.value_net.forward(x)
        return x


class LamaCritic(nn.Module):
    def __init__(
            self,
            state_shape,
            action_size,
            hiddens,
            layer_fn,
            concat_at=1,
            n_atoms=1,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=None
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        self.n_atoms = n_atoms
        state_size = state_shape[-1]  # reduce(lambda x, y: x * y, state_shape)

        if concat_at > 0:
            hiddens_ = [state_size] + hiddens[0:concat_at]
            self.observation_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )
            hiddens_ = \
                [hiddens[concat_at - 1] + action_size] + hiddens[concat_at:]
            self.feature_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )
        else:
            self.observation_net = None
            hiddens_ = [state_size + action_size] + hiddens
            self.feature_net = SequentialNet(
                hiddens=hiddens_,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias
            )

        attn_features = hiddens[-1]
        self.attn = nn.Sequential(
            nn.Conv1d(
                in_channels=attn_features,
                out_channels=1,
                kernel_size=1,
                bias=True
            ), nn.Softmax(dim=1)
        )
        self.feature_net2 = SequentialNet(
            hiddens=[hiddens[-1] * 4, hiddens[-1]],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.value_net = SequentialNet(
            hiddens=[hiddens[-1], n_atoms],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        if self.observation_net is not None:
            self.observation_net.apply(inner_init)
        self.feature_net.apply(inner_init)
        self.feature_net2.apply(inner_init)
        self.attn.apply(outer_init)
        self.value_net.apply(outer_init)

    def forward(self, states, action):
        if len(states.shape) < 3:
            states = states.unsqueeze(1)
        if self.observation_net is not None:
            states = self.observation_net(states)
        bs, ln, fd = states.shape
        actions = torch.cat([action.unsqueeze(1)] * ln, dim=1)
        x = torch.cat((states, actions), dim=2)
        bs, ln, fd = x.shape
        x = x.view(-1, fd)
        x = self.feature_net(x)

        x = x.view(bs, ln, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attn(x_a) * x_a).transpose(1, 2)

        x_avg = x.mean(1, keepdim=True)
        x_max = x.max(1, keepdim=True)[0]
        x_attn = x_attn.mean(1, keepdim=True)
        x_last = x[:, -1:, :]
        x = torch.cat([x_last, x_avg, x_max, x_attn], dim=1)
        x = x.view(bs, -1)

        x = self.feature_net2(x)
        x = self.value_net(x)
        return x


class ValueCritic(nn.Module):
    """ Critic which learns value function V(s).
    """

    def __init__(
            self,
            state_shape,
            hiddens,
            layer_fn,
            n_atoms=1,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=None
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn
        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        self.n_atoms = n_atoms

        state_size = reduce(lambda x, y: x * y, state_shape)

        hiddens_ = [state_size] + hiddens
        self.feature_net = SequentialNet(
            hiddens=hiddens_,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )

        self.value_net = SequentialNet(
            hiddens=[hiddens[-1], n_atoms],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.value_net.apply(outer_init)

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.feature_net.forward(x)
        x = self.value_net.forward(x)
        return x


class LamaValueCritic(nn.Module):
    def __init__(
            self,
            state_shape,
            hiddens,
            layer_fn,
            n_atoms=1,
            activation_fn=nn.ReLU,
            norm_fn=None,
            bias=True,
            out_activation=None
    ):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)
        self.n_atoms = n_atoms
        state_size = state_shape[-1]

        self.feature_net = SequentialNet(
            hiddens=[state_size] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.attn = nn.Sequential(
            nn.Conv1d(
                in_channels=hiddens[-1],
                out_channels=1,
                kernel_size=1,
                bias=True
            ), nn.Softmax(dim=1)
        )
        self.feature_net2 = SequentialNet(
            hiddens=[hiddens[-1] * 4, hiddens[-1]],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias
        )
        self.value_net = SequentialNet(
            hiddens=[hiddens[-1], n_atoms],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.feature_net.apply(inner_init)
        self.attn.apply(outer_init)
        self.feature_net2.apply(inner_init)
        self.value_net.apply(outer_init)

    def forward(self, states):
        if len(states.shape) < 3:
            states = states.unsqueeze(1)
        bs, ln, fd = states.shape
        x = states.view(-1, fd)
        x = self.feature_net(x)

        x = x.view(bs, ln, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attn(x_a) * x_a).transpose(1, 2)

        x_avg = x.mean(1, keepdim=True)
        x_max = x.max(1, keepdim=True)[0]
        x_attn = x_attn.mean(1, keepdim=True)
        x_last = x[:, -1:, :]
        x = torch.cat([x_last, x_avg, x_max, x_attn], dim=1)
        x = x.view(bs, -1)

        x = self.feature_net2(x)
        x = self.value_net(x)
        return x
