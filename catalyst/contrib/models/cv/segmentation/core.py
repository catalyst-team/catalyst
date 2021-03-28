# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, Union
from pathlib import Path

import torch
from torch import nn

from catalyst.contrib.models.cv.segmentation.bridge import BridgeSpec
from catalyst.contrib.models.cv.segmentation.decoder import DecoderSpec
from catalyst.contrib.models.cv.segmentation.encoder import EncoderSpec, ResnetEncoder, UnetEncoder
from catalyst.contrib.models.cv.segmentation.head import HeadSpec


class UnetMetaSpec(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        encoder: EncoderSpec,
        decoder: DecoderSpec,
        bridge: BridgeSpec = None,
        head: HeadSpec = None,
        state_dict: Union[dict, str, Path] = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge or (lambda x: x)
        self.decoder = decoder
        self.head = head or (lambda x: x)

        if state_dict is not None:
            if isinstance(state_dict, (Path, str)):
                state_dict = torch.load(str(state_dict))
            if "model_state_dict" in state_dict.keys():
                state_dict = state_dict["model_state_dict"]
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        encoder_features: List[torch.Tensor] = self.encoder(x)
        bridge_features: List[torch.Tensor] = self.bridge(encoder_features)
        decoder_features: List[torch.Tensor] = self.decoder(bridge_features)
        output: torch.Tensor = self.head(decoder_features)
        return output


class UnetSpec(UnetMetaSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 4,
        encoder_params: Dict = None,
        bridge_params: Dict = None,
        decoder_params: Dict = None,
        head_params: Dict = None,
        state_dict: Union[dict, str, Path] = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        encoder_params = encoder_params or {}
        bridge_params = bridge_params or {}
        decoder_params = decoder_params or {}
        head_params = head_params or {}

        encoder = UnetEncoder(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks,
            **encoder_params
        )

        encoder, bridge, decoder, head = self._get_components(
            encoder, num_classes, bridge_params, decoder_params, head_params
        )

        super().__init__(
            encoder=encoder, bridge=bridge, decoder=decoder, head=head, state_dict=state_dict,
        )

    def _get_components(
        self,
        encoder: UnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        raise NotImplementedError()


class ResnetUnetSpec(UnetMetaSpec):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        num_classes: int = 1,
        arch: str = "resnet18",
        pretrained: bool = True,
        encoder_params: Dict = None,
        bridge_params: Dict = None,
        decoder_params: Dict = None,
        head_params: Dict = None,
        state_dict: Union[dict, str, Path] = None,
    ):
        """@TODO: Docs. Contribution is welcome."""
        encoder_params = encoder_params or {}
        bridge_params = bridge_params or {}
        decoder_params = decoder_params or {}
        head_params = head_params or {}

        encoder = ResnetEncoder(arch=arch, pretrained=pretrained, **encoder_params)

        encoder, bridge, decoder, head = self._get_components(
            encoder, num_classes, bridge_params, decoder_params, head_params
        )

        super().__init__(
            encoder=encoder, bridge=bridge, decoder=decoder, head=head, state_dict=state_dict,
        )

    def _get_components(
        self,
        encoder: UnetEncoder,
        num_classes: int,
        bridge_params: Dict,
        decoder_params: Dict,
        head_params: Dict,
    ):
        raise NotImplementedError()


__all__ = ["UnetMetaSpec", "UnetSpec", "ResnetUnetSpec"]
