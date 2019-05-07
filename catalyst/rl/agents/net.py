import torch
import torch.nn as nn
from catalyst.contrib.models import SequentialNet
from .layers import LamaPooling


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
