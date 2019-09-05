from typing import Tuple, List, Dict, Union
from functools import partial, reduce
from copy import deepcopy
import collections

import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.contrib.modules import LamaPooling
from catalyst import utils

from ipdb import set_trace as st


def _get_network_output(net: nn.Module, input_shape: Tuple):
    input_t = torch.Tensor(torch.randn((1,) + input_shape))
    output_t = net(input_t)
    return output_t


def _get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    kernel_sizes: List = None,
    strides: List = None,
    groups: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    dropout_rate: float = None,
    activation: str = "ReLU"
) -> nn.Module:

    channels = channels or [16, 32, 16]
    kernel_sizes = kernel_sizes or [8, 4, 3]
    strides = strides or [4, 2, 1]
    groups = groups or [1, 1, 1]
    activation_fn = torch.nn.__dict__[activation]
    assert len(channels) == len(kernel_sizes) == len(strides) == len(groups)

    def _get_block(**conv_params):
        layers = [nn.Conv2d(**conv_params)]
        if use_normalization:
            layers.append(nn.InstanceNorm2d(conv_params["out_channels"]))
        if dropout_rate is not None:
            layers.append(nn.Dropout2d(p=dropout_rate))
        layers.append(activation_fn(inplace=True))
        return layers

    channels.insert(0, history_len * in_channels)
    params = []
    for i, (in_channels, out_channels) in enumerate(utils.pairwise(channels)):
        params.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "bias": use_bias,
                "kernel_size": kernel_sizes[i],
                "stride": strides[i],
                "groups": groups[i],
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    dropout_rate: float = None,
    activation: str = "ReLU",
    residual: Union[bool, str] = False,
    layer_order: List = None,
) -> nn.Module:

    features = features or [64, 128, 64]
    features.insert(0, history_len * in_features)

    net = SequentialNet(
        hiddens=features,
        layer_fn=nn.Linear,
        bias=use_bias,
        norm_fn=nn.LayerNorm if use_normalization else None,
        dropout=partial(nn.Dropout, p=dropout_rate) \
            if dropout_rate is not None \
            else None,
        activation_fn=activation,
        residual=residual,
        layer_order=layer_order,
    )

    return net


def _get_observation_net(state_shape, **observation_net_params):
    observation_net_params = deepcopy(observation_net_params)
    observation_net_type = \
        observation_net_params.pop("_network_type", "linear")

    if observation_net_type == "linear":
        # 0 - history len
        observation_size = reduce(
            lambda x, y: x * y, state_shape[1:])
        observation_net_params["in_features"] = observation_size
        observation_net = _get_linear_net(**observation_net_params)
    elif observation_net_type == "convolution":
        # 0 - history len
        observation_net_params["in_channels"] = state_shape[1]
        observation_net = _get_convolution_net(
            **observation_net_params)
    else:
        raise NotImplementedError()

    return observation_net


def _get_observation_net_out_features(
    observation_net,
    state_shape,
    **observation_net_params
):
    if observation_net_params.get("history_len", 1) == 1:
        # we need to process each observation separately
        state_shape_ = state_shape[1:]
        observation_net_output: torch.Tensor = \
            _get_network_output(
                observation_net,
                state_shape_
            )
    else:
        # we need to stack observations
        state_shape_ = (state_shape[0] * state_shape[1], ) + state_shape[2:]
        observation_net_output: torch.Tensor = \
            _get_network_output(
                observation_net,
                state_shape_
            )
    result = observation_net_output.nelement()
    return result


def get_observation_net(state_shape, **observation_net_params):
    if len(observation_net_params) == 0:
        # no observation net required
        observation_net = None
        observation_net_out_features = 0
        if isinstance(state_shape, dict):
            for value in state_shape.values():
                observation_net_out_features += reduce(
                    lambda x, y: x * y, state_shape)
        else:
            observation_net_out_features = reduce(
                lambda x, y: x * y, state_shape)
    else:
        observation_net: nn.Module = \
            _get_observation_net(
                state_shape,
                **observation_net_params
            )
        observation_net_out_features = \
            _get_observation_net_out_features(
                observation_net,
                state_shape,
                **observation_net_params
            )
    return observation_net, observation_net_out_features


class StateNet(nn.Module):
    @staticmethod
    def _encode_state_ff(state: torch.Tensor, observation_net: nn.Module):
        x = state

        if len(x.shape) == 5:  # image input
            x = x / 255.
            batch_size, history_len, c, h, w = x.shape

            x = x.view(batch_size, -1, h, w)
            x = observation_net(x)
        else:  # vector input
            batch_size, history_len, f = x.shape
            x = x.view(batch_size, -1)
            x = observation_net(x)

        x = x.view(batch_size, -1)
        return x

    @staticmethod
    def _encode_state_ff_kv(state: Dict, observation_net: nn.ModuleDict):
        x: List[torch.Tensor] = []
        for key, net in observation_net.items():
            x.append(StateNet._encode_state_ff(state[key], net))
        x = torch.cat(x, dim=-1)
        return x

    @staticmethod
    def _encode_state_lama(state: torch.Tensor, observation_net: nn.Module):
        x = state

        if len(x.shape) == 5:  # image input
            x = x / 255.
            batch_size, history_len, c, h, w = x.shape

            x = x.view(-1, c, h, w)
            x = observation_net(x)
        else:  # vector input
            batch_size, history_len, f = x.shape
            x = x.view(-1, f)
            x = observation_net(x)

        x = x.view(batch_size, history_len, -1)
        return x

    @staticmethod
    def _encode_state_lama_kv(state: Dict, observation_net: nn.ModuleDict):
        x: List[torch.Tensor] = []
        for key, net in observation_net.items():
            x.append(StateNet._encode_state_lama(state[key], net))
        x = torch.cat(x, dim=-1)
        return x

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
            self._process_state = StateNet._encode_state_ff_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else StateNet._encode_state_ff
        elif isinstance(aggregation_net, LamaPooling):
            self._forward_fn = self._forward_lama
            self._process_state = StateNet._encode_state_lama_kv \
                if isinstance(self.observation_net, nn.ModuleDict) \
                else StateNet._encode_state_lama
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
                        get_observation_net(
                            state_shape[key],
                            **observation_net_params[key]
                        )
                    observation_net[key] = net_
                    observation_net_out_features += out_features_
                observation_net = nn.ModuleDict(observation_net)
            else:
                observation_net, observation_net_out_features = \
                    get_observation_net(
                        state_shape,
                        **observation_net_params
                    )
            main_net_in_features += observation_net_out_features
        else:
            observation_net, observation_net_out_features = \
                get_observation_net(state_shape)
            main_net_in_features += observation_net_out_features

        # aggregation net
        if aggregation_net_params is not None:
            aggregation_net = LamaPooling(
                observation_net_out_features,
                **aggregation_net_params)
            main_net_in_features = aggregation_net.features_out
        else:
            aggregation_net = None

        # main net
        main_net_params["in_features"] = main_net_in_features
        main_net = _get_linear_net(**main_net_params)

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
        state_shape,
        action_shape,
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


__all__ = ["StateNet", "StateActionNet"]
