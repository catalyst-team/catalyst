# Author: Sergey Kolesnikov, scitator@gmail.com
# flake8: noqa
# @TODO: code formatting issue for 20.07 release

from typing import Dict, Union
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from catalyst import utils
from catalyst.contrib.models.sequential import SequentialNet
from catalyst.contrib.nn.modules import Normalize


class Hydra(nn.Module):
    """Hydra - one model to predict them all.

    @TODO: Docs. Contribution is welcome.
    """

    parent_keyword = "_"
    hidden_keyword = "_hidden"
    normalize_keyword = "normalize_output"

    def __init__(
        self, heads: nn.ModuleDict, encoder: nn.Module = None, embedders: nn.ModuleDict = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.encoder = encoder or nn.Sequential()
        self.heads = heads
        self.embedders = embedders or {}

    @staticmethod
    def parse_head_params(
        head_params: Dict, in_features: int, is_leaf: bool = False,
    ) -> Union[nn.Module, nn.ModuleDict]:
        """@TODO: Docs. Contribution is welcome."""
        if is_leaf:
            if isinstance(head_params, int):
                head_params = {"hiddens": [head_params]}
            normalize = head_params.pop(Hydra.normalize_keyword, False)
            head_params["hiddens"].insert(0, in_features)

            output = [("net", SequentialNet(**head_params))]
            if normalize:
                output.append(("normalize", Normalize()))

            output = OrderedDict(output)
            output = nn.Sequential(output)
        else:
            output = {}

            hidden_params = head_params.pop(Hydra.hidden_keyword, None)
            if hidden_params is not None:
                in_features = (
                    hidden_params
                    if isinstance(hidden_params, int)
                    else hidden_params["hiddens"][-1]
                )
                output[Hydra.hidden_keyword] = Hydra.parse_head_params(
                    head_params=hidden_params, in_features=in_features, is_leaf=True,
                )

            for head_branch_name, head_branch_params in head_params.items():
                output[head_branch_name] = Hydra.parse_head_params(
                    head_params=head_branch_params,
                    in_features=in_features,
                    is_leaf=not head_branch_name.startswith(Hydra.parent_keyword),
                )
            output = nn.ModuleDict(output)
        return output

    @staticmethod
    def forward_head(head_input, head):
        """@TODO: Docs. Contribution is welcome."""
        if isinstance(head, nn.ModuleDict):
            output = {}

            net = getattr(head, Hydra.hidden_keyword, None)
            if net is not None:
                head_input = net(head_input)
                output[""] = head_input

            for head_name, head_layer in head.items():
                if head_name == Hydra.hidden_keyword:
                    continue
                output[head_name] = Hydra.forward_head(head_input, head_layer)
        elif isinstance(head, nn.Module):
            output = head(head_input)
        else:
            raise NotImplementedError()
        return output

    def forward(
        self, features: torch.Tensor, **targets_kwargs,
    ):
        """Forward call."""
        embeddings = self.encoder(features)

        heads_output = self.forward_head(embeddings, self.heads)
        output = {
            "features": features,
            "embeddings": embeddings,
            **utils.flatten_dict(heads_output),
        }

        for key, value in targets_kwargs.items():
            output[f"{key}_embeddings"] = self.embedders[key](value)

        return output

    def forward_tuple(self, features: torch.Tensor):
        """@TODO: Docs. Contribution is welcome."""
        output_kv = self.forward(features)
        output = [
            output_kv["features"],
            output_kv["embeddings"],
        ]

        # let's remove all hidden parts from prediction
        output.extend(
            [
                value
                for key, value in output_kv.items()
                if not key.endswith("/") and key not in ["features", "embeddings"]
            ]
        )
        output = tuple(output)
        return output

    @classmethod
    def get_from_params(
        cls,
        heads_params: Dict,
        encoder_params: Dict = None,
        embedders_params: Dict = None,
        in_features: int = None,
    ) -> "Hydra":
        """@TODO: Docs. Contribution is welcome."""
        heads_params_copy = deepcopy(heads_params)
        encoder_params_copy = deepcopy(encoder_params)
        embedders_params_copy = deepcopy(embedders_params)

        def _get_normalization_keyword(dct: Dict):
            return dct.pop(Hydra.normalize_keyword, False) if dct is not None else False

        if encoder_params_copy is not None:
            normalize_embeddings: bool = _get_normalization_keyword(encoder_params_copy)

            encoder = SequentialNet(**encoder_params_copy)
            in_features = encoder_params_copy["hiddens"][-1]

            if normalize_embeddings:
                encoder = nn.Sequential(encoder, Normalize())
        else:
            assert in_features is not None
            encoder = None

        heads = Hydra.parse_head_params(head_params=heads_params_copy, in_features=in_features)
        assert isinstance(heads, nn.ModuleDict)

        embedders = {}
        if embedders_params_copy is not None:
            for key, head_params in embedders_params_copy.items():
                if isinstance(head_params, int):
                    head_params = {"num_embeddings": head_params}
                need_normalize = head_params.pop(Hydra.normalize_keyword, False)
                block = [("embedding", nn.Embedding(embedding_dim=in_features, **head_params))]
                if need_normalize:
                    block.append(("normalize", Normalize()))

                block = OrderedDict(block)
                block = nn.Sequential(block)
                embedders[key] = block
            embedders = nn.ModuleDict(embedders)

        net = cls(heads=heads, encoder=encoder, embedders=embedders)

        return net


__all__ = ["Hydra"]
