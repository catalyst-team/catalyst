import torch
import torch.nn as nn
from functools import reduce

from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.rl.agents.utils import normal_sample, normal_log_prob
from catalyst.rl.agents.layers import StateNet, SquashingLayer, CouplingLayer

# log_sigma of Gaussian policy are capped at (LOG_SIG_MIN, LOG_SIG_MAX)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


class Actor(StateNet):
    """
    Actor which learns deterministic policy.
    """

    @classmethod
    def create_from_config(
        cls,
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
        **kwargs
    ):
        assert len(kwargs) == 0
        # hack to prevent cycle imports
        from catalyst.contrib.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        out_activation = name2nn(out_activation)

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        if len(state_shape) in [1, 2]:
            # linear case: one observation or several one
            state_size = reduce(lambda x, y: x * y, state_shape)

            observation_net = SequentialNet(
                hiddens=[state_size] + hiddens,
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

        head_net = SequentialNet(
            hiddens=[memory_out, action_size],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        observation_net.apply(inner_init)
        head_net.apply(outer_init)

        actor_net = cls(
            observation_net=observation_net,
            memory_net=memory_net,
            head_net=head_net
        )

        return actor_net


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
