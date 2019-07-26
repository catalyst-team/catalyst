import torch
import torch.nn as nn

from catalyst.contrib.modules import LamaPooling
from catalyst import utils


def _get_observation_net(
    history_len: int = 1,
    conv1_size: int = 32,
    conv2_size: int = 64,
    conv3_size: int = 32,
    use_bias: bool = False,
    use_groups: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**conv_params):
        layers = [nn.Conv2d(**conv_params)]
        if use_normalization:
            layers.append(nn.InstanceNorm2d(conv_params["out_channels"]))
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    params = [
        {
            "in_channels": history_len * 1,
            "out_channels": conv1_size,
            "bias": use_bias,
            "kernel_size": 8,
            "stride": 4,
            "groups": history_len if use_groups else 1,
        },
        {
            "in_channels": conv1_size,
            "out_channels": conv2_size,
            "bias": use_bias,
            "kernel_size": 4,
            "stride": 2,
            "groups": 4 if use_groups else 1,
        },
        {
            "in_channels": conv2_size,
            "out_channels": conv3_size,
            "bias": use_bias,
            "kernel_size": 3,
            "stride": 1,
            "groups": 4 if use_groups else 1,
        },
    ]

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    # input_shape: tuple = (1, 84, 84)
    # conv_input = torch.Tensor(torch.randn((1,) + input_shape))
    # conv_output = net(conv_input)
    # torch.Size([1, 32, 7, 7]), 1568
    # print(conv_output.shape, conv_output.nelement())

    return net


def _get_ff_main_net(
    in_features: int,
    out_features: int,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    activation_fn = torch.nn.__dict__[activation]

    layers = [nn.Linear(in_features, out_features, bias=use_bias)]
    if use_normalization:
        layers.append(nn.LayerNorm(out_features))
    if use_dropout:
        layers.append(nn.Dropout(p=0.1))
    layers.append(activation_fn(inplace=True))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


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
        self.observation_net = observation_net
        self.aggregation_net = aggregation_net

        self._forward_fn = None
        if aggregation_net is None:
            self._forward_fn = self._forward_ff
        elif isinstance(aggregation_net, LamaPooling):
            self._forward_fn = self._forward_lama
        else:
            raise NotImplementedError

    def _forward_ff(self, state):
        x = state

        x = x / 255.
        batch_size, history_len, c, h, w = x.shape

        x = x.view(batch_size, -1, h, w)
        x = self.observation_net(x)

        x = x.view(batch_size, -1)
        x = self.main_net(x)
        return x

    def _forward_lama(self, state):
        x = state

        x = x / 255.
        batch_size, history_len, c, h, w = x.shape

        x = x.view(-1, c, h, w)
        x = self.observation_net(x)

        x = x.view(batch_size, history_len, -1)
        x = self.aggregation_net(x)

        x = x.view(batch_size, -1)
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

        observation_net = _get_observation_net(**observation_net_params)
        observation_net_out_features = \
            observation_net_params["conv3_size"] * 7 * 7

        if aggregation_net_params is None:
            aggregation_net = None
            main_net_in_features = observation_net_out_features
        else:
            aggregation_net = LamaPooling(
                observation_net_out_features, **aggregation_net_params)
            main_net_in_features = aggregation_net.features_out

        main_net_params["in_features"] = main_net_in_features
        main_net = _get_ff_main_net(**main_net_params)

        net = cls(
            observation_net=observation_net,
            aggregation_net=aggregation_net,
            main_net=main_net
        )
        return net
