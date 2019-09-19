import torch
from torch import nn
from catalyst.utils import outer_init


class TemporalLastPooling(nn.Module):
    def forward(self, x):
        x_out = x[:, -1:, :]
        return x_out


class TemporalAvgPooling(nn.Module):
    def forward(self, x):
        x_out = x.mean(1, keepdim=True)
        return x_out


class TemporalMaxPooling(nn.Module):
    def forward(self, x):
        x_out = x.max(1, keepdim=True)[0]
        return x_out


class TemporalAttentionPooling(nn.Module):
    name2activation = {
        "softmax": nn.Softmax(dim=1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }

    def __init__(self, features_in, activation=None, kernel_size=1, **params):
        super().__init__()
        self.features_in = features_in
        activation = activation or "softmax"

        self.attention_pooling = nn.Sequential(
            nn.Conv1d(
                in_channels=features_in,
                out_channels=1,
                kernel_size=kernel_size,
                **params
            ),
            TemporalAttentionPooling.name2activation[activation]
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


class TemporalConcatPooling(nn.Module):
    def __init__(self, features_in, history_len=1):
        super().__init__()
        self.features_in = features_in
        self.features_out = features_in * history_len

    def forward(self, x):
        """
        :param x: [batch_size, history_len, feature_size]
        :return:
        """
        x = x.view(x.shape[0], -1)
        return x


class TemporalDropLastWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = x[:, :-1, :]
        x_out = self.net(x)
        return x_out


def get_pooling(key, features_in, **params):
    key_ = key.split("_", 1)[0]

    if key_ == "last":
        return TemporalLastPooling()
    elif key_ == "avg":
        layer = TemporalAvgPooling()
    elif key_ == "max":
        layer = TemporalMaxPooling()
    elif key_ in ["softmax", "tanh", "sigmoid"]:
        layer = TemporalAttentionPooling(
            features_in=features_in, activation=key_, **params)
    else:
        raise NotImplementedError()

    if "droplast" in key:
        layer = TemporalDropLastWrapper(layer)

    return layer


class LamaPooling(nn.Module):
    available_groups = [
        "last",
        "avg", "avg_droplast",
        "max", "max_droplast",
        "sigmoid", "sigmoid_droplast",
        "softmax", "softmax_droplast",
        "tanh", "tanh_droplast",
    ]

    def __init__(self, features_in, groups=None):
        super().__init__()
        self.features_in = features_in
        self.groups = groups \
            or ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
        self.features_out = features_in * len(self.groups)

        groups = {}
        for key in self.groups:
            if isinstance(key, str):
                groups[key] = get_pooling(key, self.features_in)
            elif isinstance(key, dict):
                key_ = key.pop("key")
                groups[key_] = get_pooling(key_, features_in, **key)
            else:
                raise NotImplementedError()

        self.groups = nn.ModuleDict(groups)

    def forward(self, x):
        """
        :param x: [batch_size, history_len, feature_size]
        :return:
        """
        batch_size, history_len, feature_size = x.shape

        x_ = []
        for pooling_fn in self.groups.values():
            features_ = pooling_fn(x)
            x_.append(features_)
        x = torch.cat(x_, dim=1)
        x = x.view(batch_size, -1)

        return x


__all__ = [
    "TemporalLastPooling",
    "TemporalAvgPooling",
    "TemporalMaxPooling",
    "TemporalDropLastWrapper",
    "TemporalAttentionPooling",
    "TemporalConcatPooling",
    "LamaPooling",
]
