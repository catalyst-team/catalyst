from typing import Dict, Optional, Sequence, Tuple, Union

from kornia.augmentation import AugmentationBase
import torch
from torch import nn

from catalyst.contrib.registry import TRANSFORMS
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class BatchTransformCallback(Callback):
    """Callback to perform data augmentations on GPU using kornia library.

    Look at `Kornia: an Open Source Differentiable Computer Vision
    Library for PyTorch`_ for details.

    .. _`Kornia: an Open Source Differentiable Computer Vision Library
        for PyTorch`: https://arxiv.org/pdf/1910.02190.pdf
    """

    def __init__(
        self,
        transform: Sequence[dict],
        input_key: str,
        output_key: Optional[str] = None,
        additional_input_key: Optional[str] = None,
        additional_output_key: Optional[str] = None,
        loader: Optional[str] = None,
    ) -> None:
        """Constructor method for the :class:`BatchTransformCallback` callback.

        Args:
            transform (Sequence[Union[dict, AugmentationBase]]): A sequence
                of dits with params for each kornia transform or sequence of
                transforms to apply. If sequence of params then must contain
                `transform` key with augmentation name as a value
                and if augmentation is custom, then you should add it
                to the `TRANSFORMS` registry first.
            input_key (str): Key in batch dict mapping to to tranform,
                e.g. `'image'`.
            output_key (Optional[str]): Key to use to store the result
                of transform. Defaults to `input_key` if not provided.
            additional_input_key (Optional[str]): Key of additional target
                in batch dict mapping to to tranform, e.g. `'mask'`.
            additional_output_key (Optional[str]): Key to use to store
                the result of additional target transform.
                Defaults to `additional_input_key` if not provided.
            loader (Optional[str]): Name of the loader on which items
                transform should be applied. If `None`, transform going to be
                applied for each loader.
        """
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.all)

        self.input_key = input_key
        self.additional_input = additional_input_key
        self._process_input = (
            self._process_input_tuple
            if self.additional_input is not None
            else self._process_input_tensor
        )

        self.output_key = output_key or input_key
        self.additional_output = additional_output_key or self.additional_input
        self._process_output = (
            self._process_output_tuple
            if self.additional_output is not None
            else self._process_output_tensor
        )

        transforms: Sequence[AugmentationBase] = [
            item
            if isinstance(item, AugmentationBase)
            else TRANSFORMS.get_from_params(**item)
            for item in transform
        ]
        assert all(
            isinstance(t, AugmentationBase) for t in transforms
        ), "`kornia.AugmentationBase` should be a base class for transforms"

        self.transform = nn.Sequential(*transforms)
        self.loader = loader

    def _process_input_tensor(self, input_: dict) -> torch.Tensor:
        return input_[self.input_key]

    def _process_input_tuple(
        self, input_: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return input_[self.input_key], input_[self.additional_input]

    def _process_output_tensor(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {self.output_key: batch}

    def _process_output_tuple(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out_t, additional_t = batch
        return {self.output_key: out_t, self.additional_output: additional_t}

    def on_batch_start(self, runner: IRunner) -> None:
        """Apply transforms.

        Args:
            runner (IRunner): Current runner.
        """
        if self.loader is None or runner.loader_name == self.loader:
            in_batch = self._process_input(runner.input)
            out_batch = self.transform(in_batch)
            runner.input.update(self._process_output(out_batch))


__all__ = ["BatchTransformCallback"]
