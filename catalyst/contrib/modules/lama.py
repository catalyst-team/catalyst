import torch
from torch import nn
from catalyst.utils import outer_init


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
            k: self._get_pooling(k, self.features_in)
            for k in self.poolings
        })

    @staticmethod
    def _get_pooling(key, features_in):
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


__all__ = ["TemporalAttentionPooling", "LamaPooling"]
