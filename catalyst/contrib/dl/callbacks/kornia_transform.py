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

    To apply augmentations only during specific loader e.g. only during
    training :class:`catalyst.core.callbacks.control_flow.ControlFlowCallback`
    callback can be used. For config API it can look like this:

    .. code-block:: yaml

        callbacks_params:
          ...
          train_transforms:
            _wrapper:
              callback: ControlFlowCallback
              loaders: train
            callback: BatchTransformCallback
            transforms:
              - transform: kornia.RandomAffine
                degrees: [-15, 20]
                scale: [0.75, 1.25]
                return_transform: true
              - transform: kornia.ColorJitter
                brightness: 0.1
                contrast: 0.1
                saturation: 0.1
                return_transform: false
            input_key: image
            additional_input_key: mask
          ...

    .. _`Kornia: an Open Source Differentiable Computer Vision Library
        for PyTorch`: https://arxiv.org/pdf/1910.02190.pdf
    """

    def __init__(
        self,
        transform: Sequence[Union[dict, AugmentationBase]],
        input_key: str = "image",
        additional_input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        additional_output_key: Optional[str] = None,
    ) -> None:
        """Constructor method for the :class:`BatchTransformCallback` callback.

        Args:
            transform (Sequence[Union[dict, AugmentationBase]]): define
                augmentations to apply on a batch

                If a sequence of transforms passed, then each element
                should be either ``kornia.augmentation.AugmentationBase`` or
                ``nn.Module`` compatible with kornia interface.

                If a sequence of params (``dict``) passed, then each
                element of the sequence must contain ``'transform'`` key with
                an augmentation name as a value. Please note that in this case
                to use custom augmentation you should add it to the
                `TRANSFORMS` registry first.
            input_key (str): key in batch dict mapping to transform,
                e.g. `'image'`
            additional_input_key (Optional[str]): key of an additional target
                in batch dict mapping to transform, e.g. `'mask'`
            output_key (Optional[str]): key to use to store the result
                of the transform, defaults to `input_key` if not provided
            additional_output_key (Optional[str]): key to use to store
                the result of additional target transformation,
                defaults to `additional_input_key` if not provided
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
            runner (IRunner): сurrent runner
        """
        in_batch = self._process_input(runner.input)
        out_batch = self.transform(in_batch)
        runner.input.update(self._process_output(out_batch))


__all__ = ["BatchTransformCallback"]
