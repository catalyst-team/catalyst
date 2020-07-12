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

    Usage example for notebook API:

    .. code-block:: python

        import os

        from kornia import augmentation
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader

        from catalyst import dl
        from catalyst.contrib.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST
        from catalyst.contrib.dl.callbacks.kornia_transform import (
            BatchTransformCallback
        )
        from catalyst.utils import metrics


        class CustomRunner(dl.Runner):
            def predict_batch(self, batch):
                # model inference step
                return self.model(
                    batch[0].to(self.device).view(batch[0].size(0), -1)
                )

            def _handle_batch(self, batch):
                # model train/valid step
                x, y = batch
                y_hat = self.model(x.view(x.size(0), -1))

                loss = F.cross_entropy(y_hat, y)
                accuracy01, *_ = metrics.accuracy(y_hat, y)
                self.batch_metrics.update(
                    {"loss": loss, "accuracy01": accuracy01}
                )

                if self.is_train_loader:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        model = torch.nn.Linear(28 * 28, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, transform=ToTensor()),
                batch_size=32,
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, transform=ToTensor()),
                batch_size=32,
            ),
        }
        transrorms = [
            augmentation.RandomAffine(degrees=(-15, 20), scale=(0.75, 1.25)),
        ]

        runner = CustomRunner()

        # model training
        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            logdir="./logs",
            num_epochs=5,
            verbose=True,
            callbacks=[BatchTransformCallback(transrorms, input_key=0)],
        )

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
        input_key: Union[str, int] = "image",
        additional_input_key: Optional[str] = None,
        output_key: Optional[Union[str, int]] = None,
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
            input_key (Union[str, int]): key in batch dict
                mapping to transform, e.g. `'image'`
            additional_input_key (Optional[Union[str, int]]): key of an
                additional target in batch dict mapping to transform,
                e.g. `'mask'`
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
        self, runner: IRunner, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        runner.input[self.output_key] = batch

    def _process_output_tuple(
        self, runner: IRunner, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        out_t, additional_t = batch
        dict_ = {self.output_key: out_t, self.additional_output: additional_t}
        runner.input.update(dict_)

    def on_batch_start(self, runner: IRunner) -> None:
        """Apply transforms.

        Args:
            runner (IRunner): сurrent runner
        """
        in_batch = self._process_input(runner.input)
        out_batch = self.transform(in_batch)
        self._process_output(runner, out_batch)


__all__ = ["BatchTransformCallback"]
