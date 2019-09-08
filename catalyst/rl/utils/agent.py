from typing import List, Dict
from functools import reduce
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst.contrib.models import get_convolution_net, get_linear_net
from catalyst import utils


def _get_observation_net(state_shape, **observation_net_params):
    # @TODO: make more general and move to contrib
    observation_net_params = deepcopy(observation_net_params)
    observation_net_type = \
        observation_net_params.pop("_network_type", "linear")

    if observation_net_type == "linear":
        # 0 - history len
        observation_size = reduce(
            lambda x, y: x * y, state_shape[1:])
        observation_net_params["in_features"] = observation_size
        observation_net = get_linear_net(**observation_net_params)
    elif observation_net_type == "convolution":
        # 0 - history len
        observation_net_params["in_channels"] = state_shape[1]
        observation_net = get_convolution_net(
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
            utils.get_network_output(
                observation_net,
                state_shape_
            )
    else:
        # we need to stack observations
        state_shape_ = (state_shape[0] * state_shape[1], ) + state_shape[2:]
        observation_net_output: torch.Tensor = \
            utils.get_network_output(
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


def process_state_ff(state: torch.Tensor, observation_net: nn.Module):
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


def process_state_ff_kv(state: Dict, observation_net: nn.ModuleDict):
    x: List[torch.Tensor] = []
    for key, net in observation_net.items():
        x.append(process_state_ff(state[key], net))
    x = torch.cat(x, dim=-1)
    return x


def process_state_temporal(state: torch.Tensor, observation_net: nn.Module):
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


def process_state_temporal_kv(state: Dict, observation_net: nn.ModuleDict):
    x: List[torch.Tensor] = []
    for key, net in observation_net.items():
        x.append(process_state_temporal(state[key], net))
    x = torch.cat(x, dim=-1)
    return x
