import collections

import torch
import torch.nn as nn

from catalyst.contrib.models import get_linear_net
from catalyst.contrib.modules import LamaPooling, TemporalConcatPooling
from catalyst.rl import utils


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        observation_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        """
        Abstract network, that takes some tensor
        T of shape [bs; history_len; ...]
        and outputs some representation tensor R
        of shape [bs; representation_size]

        input_T [bs; history_len; in_features]

        -> observation_net (aka observation_encoder) ->

        observations_representations [bs; history_len; obs_features]

        -> aggregation_net (flatten in simplified case) ->

        aggregated_representation [bs; hid_features]

        -> main_net ->

        output_T [bs; representation_size]

        Args:
            main_net:
            observation_net:
            aggregation_net:
        """
        super().__init__()
        self.main_net = main_net
        self.observation_net = observation_net or (lambda x: x)
        self.aggregation_net = aggregation_net

        self._forward_fn = None
        if aggregation_net is None:
            self._forward_fn = self._forward_ff
            self._process_state = utils.process_state_ff_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else utils.process_state_ff
        elif isinstance(aggregation_net, (TemporalConcatPooling, LamaPooling)):
            self._forward_fn = self._forward_lama
            self._process_state = utils.process_state_temporal_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else utils.process_state_temporal
        else:
            raise NotImplementedError()

    def _forward_ff(self, state):
        x = state
        x = self._process_state(x, self.observation_net)
        x = self.main_net(x)
        return x

    def _forward_lama(self, state):
        x = state
        x = self._process_state(x, self.observation_net)
        x = self.aggregation_net(x)
        x = self.main_net(x)
        return x

    def forward(self, state):
        x = self._forward_fn(state)
        return x

    @classmethod
    def get_from_params(
        cls,
        state_shape,
        observation_net_params=None,
        aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":

        assert main_net_params is not None
        # @TODO: refactor, too complicated; fast&furious development

        main_net_in_features = 0
        observation_net_out_features = 0

        # observation net
        if observation_net_params is not None:
            key_value_flag = observation_net_params.pop("_key_value", False)

            if key_value_flag:
                observation_net = collections.OrderedDict()
                for key in observation_net_params:
                    net_, out_features_ = \
                        utils.get_observation_net(
                            state_shape[key],
                            **observation_net_params[key]
                        )
                    observation_net[key] = net_
                    observation_net_out_features += out_features_
                observation_net = nn.ModuleDict(observation_net)
            else:
                observation_net, observation_net_out_features = \
                    utils.get_observation_net(
                        state_shape,
                        **observation_net_params
                    )
            main_net_in_features += observation_net_out_features
        else:
            observation_net, observation_net_out_features = \
                utils.get_observation_net(state_shape)
            main_net_in_features += observation_net_out_features

        # aggregation net
        if aggregation_net_params is not None:
            aggregation_type = \
                aggregation_net_params.pop("_network_type", "concat")

            if aggregation_type == "concat":
                aggregation_net = TemporalConcatPooling(
                    observation_net_out_features, **aggregation_net_params)
            elif aggregation_type == "lama":
                aggregation_net = LamaPooling(
                    observation_net_out_features,
                    **aggregation_net_params)
            else:
                raise NotImplementedError()

            main_net_in_features = aggregation_net.features_out
        else:
            aggregation_net = None

        # main net
        main_net_params["in_features"] = main_net_in_features
        main_net = get_linear_net(**main_net_params)

        net = cls(
            main_net=main_net,
            aggregation_net=aggregation_net,
            observation_net=observation_net)
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
            self._process_state = utils.process_state_ff_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else utils.process_state_ff
        elif isinstance(aggregation_net, (TemporalConcatPooling, LamaPooling)):
            self._forward_fn = self._forward_lama
            self._process_state = utils.process_state_temporal_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else utils.process_state_temporal
        else:
            raise NotImplementedError()

    def _forward_ff(self, state, action):
        state_ = self._process_state(state, self.observation_net)
        action_ = self.action_net(action)
        x = torch.cat((state_, action_), dim=1)
        x = self.main_net(x)
        return x

    def _forward_lama(self, state, action):
        state_ = self._process_state(state, self.observation_net)
        state_ = self.aggregation_net(state_)
        action_ = self.action_net(action)
        x = torch.cat((state_, action_), dim=1)
        x = self.main_net(x)
        return x

    def forward(self, state, action):
        x = self._forward_fn(state, action)
        return x

    @classmethod
    def get_from_params(
        cls,
        state_shape,
        action_shape,
        observation_net_params=None,
        action_net_params=None,
        aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        assert main_net_params is not None
        # @TODO: refactor, too complicated; fast&furious development

        main_net_in_features = 0
        observation_net_out_features = 0

        # observation net
        if observation_net_params is not None:
            key_value_flag = observation_net_params.pop("_key_value", False)

            if key_value_flag:
                observation_net = collections.OrderedDict()
                for key in observation_net_params:
                    net_, out_features_ = \
                        utils.get_observation_net(
                            state_shape[key],
                            **observation_net_params[key]
                        )
                    observation_net[key] = net_
                    observation_net_out_features += out_features_
                observation_net = nn.ModuleDict(observation_net)
            else:
                observation_net, observation_net_out_features = \
                    utils.get_observation_net(
                        state_shape,
                        **observation_net_params
                    )
        else:
            observation_net, observation_net_out_features = \
                utils.get_observation_net(state_shape)
        main_net_in_features += observation_net_out_features

        # aggregation net
        if aggregation_net_params is not None:
            aggregation_type = \
                aggregation_net_params.pop("_network_type", "concat")

            if aggregation_type == "concat":
                aggregation_net = TemporalConcatPooling(
                    observation_net_out_features, **aggregation_net_params)
            elif aggregation_type == "lama":
                aggregation_net = LamaPooling(
                    observation_net_out_features,
                    **aggregation_net_params)
            else:
                raise NotImplementedError()

            main_net_in_features = aggregation_net.features_out
        else:
            aggregation_net = None

        # action net
        if action_net_params is not None:
            # @TODO: hacky solution for code reuse
            action_shape = (1, ) + action_shape
            action_net, action_net_out_features = \
                utils.get_observation_net(action_shape, **action_net_params)
        else:
            action_net, action_net_out_features = \
                utils.get_observation_net(action_shape)
        main_net_in_features += action_net_out_features

        # main net
        main_net_params["in_features"] = main_net_in_features
        main_net = get_linear_net(**main_net_params)

        net = cls(
            observation_net=observation_net,
            action_net=action_net,
            aggregation_net=aggregation_net,
            main_net=main_net
        )
        return net


__all__ = ["StateNet", "StateActionNet"]
