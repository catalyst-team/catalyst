# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import torch
from torch import nn

from catalyst.utils.torch import outer_init


class TemporalLastPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward call."""
        x_out = x[:, -1:, :]
        return x_out


class TemporalAvgPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward call."""
        if mask is None:
            x_out = x.mean(1, keepdim=True)
        else:
            x_masked = torch.sum(x * mask.float(), dim=1, keepdim=True)
            mask_sum = torch.sum(mask.float(), dim=1, keepdim=True)
            x_out = x_masked / mask_sum
        return x_out


class TemporalMaxPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward call."""
        if mask is not None:
            x_mask = (~mask.bool()).float() * (-x.max()).float()
            x = torch.sum(x + x_mask, dim=1, keepdim=True)
        x_out = x.max(1, keepdim=True)[0]
        return x_out


class TemporalAttentionPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    name2activation = {
        "softmax": nn.Softmax(dim=1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(self, in_features, activation=None, kernel_size=1, **params):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.in_features = in_features
        activation = activation or "softmax"

        self.attention_pooling = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=1, kernel_size=kernel_size, **params),
            TemporalAttentionPooling.name2activation[activation],
        )
        self.attention_pooling.apply(outer_init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward call.

        Args:
            x: tensor of size
                (batch_size, history_len, feature_size)
            mask: mask to use

        Returns:
            pooling result
        """
        batch_size, history_len, feature_size = x.shape

        x = x.view(batch_size, history_len, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attention_pooling(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)

        return x_attn


class TemporalConcatPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, in_features, history_len=1):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features * history_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Concat pooling forward.

        Args:
            x: tensor of size
                (batch_size, history_len, feature_size)
            mask: mask to use

        Returns:
            concated result
        """
        x = x.view(x.shape[0], -1)
        return x


class TemporalDropLastWrapper(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, net):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """@TODO: Docs. Contribution is welcome."""
        x = x[:, :-1, :]
        x_out = self.net(x)
        return x_out


def _get_pooling(key, in_features, **params):
    """@TODO: Docs. Contribution is welcome."""
    key_prefix = key.split("_", 1)[0]

    if key_prefix == "last":
        return TemporalLastPooling()
    elif key_prefix == "avg":
        layer = TemporalAvgPooling()
    elif key_prefix == "max":
        layer = TemporalMaxPooling()
    elif key_prefix in ["softmax", "tanh", "sigmoid"]:
        layer = TemporalAttentionPooling(in_features=in_features, activation=key_prefix, **params)
    else:
        raise NotImplementedError()

    if "droplast" in key:
        layer = TemporalDropLastWrapper(layer)

    return layer


class LamaPooling(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    available_groups = [
        "last",
        "avg",
        "avg_droplast",
        "max",
        "max_droplast",
        "sigmoid",
        "sigmoid_droplast",
        "softmax",
        "softmax_droplast",
        "tanh",
        "tanh_droplast",
    ]

    def __init__(self, in_features, groups=None):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.in_features = in_features
        self.groups = groups or [
            "last",
            "avg_droplast",
            "max_droplast",
            "softmax_droplast",
        ]
        self.out_features = in_features * len(self.groups)

        groups = {}
        for key in self.groups:
            if isinstance(key, str):
                groups[key] = _get_pooling(key, self.in_features)
            elif isinstance(key, dict):
                key_key = key.pop("key")
                groups[key_key] = _get_pooling(key_key, in_features, **key)
            else:
                raise NotImplementedError()

        self.groups = nn.ModuleDict(groups)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward method of the LAMA.

        Args:
            x: tensor of size
                (batch_size, history_len, feature_size)
            mask: mask to use for attention compute

        Returns:
            torch.Tensor: LAMA pooling result
        """
        batch_size, history_len, feature_size = x.shape

        features_list = []
        for pooling_fn in self.groups.values():
            features = pooling_fn(x, mask)
            features_list.append(features)
        x = torch.cat(features_list, dim=1)
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
