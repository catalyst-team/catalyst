from typing import Optional, Sequence, TYPE_CHECKING, Union

from torch import nn

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.registry import REGISTRY

if TYPE_CHECKING:
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
        from catalyst.contrib.callbacks.kornia_transform import (
            BatchTransformCallback
        )
        from catalyst import metrics


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
              name: ControlFlowCallback
              loaders: train
            name: BatchTransformCallback
            transform:
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
          ...

    .. _`Kornia: an Open Source Differentiable Computer Vision Library
        for PyTorch`: https://arxiv.org/pdf/1910.02190.pdf
    """

    def __init__(
        self,
        transform: Sequence[Union[dict, nn.Module]],
        input_key: Union[str, int] = "image",
        output_key: Optional[Union[str, int]] = None,
    ) -> None:
        """Constructor method for the :class:`BatchTransformCallback` callback.

        Args:
            transform (Sequence[Union[dict, nn.Module]]): define
                augmentations to apply on a batch

                If a sequence of transforms passed, then each element
                should be either ``kornia.augmentation.AugmentationBase2D``,
                ``kornia.augmentation.AugmentationBase3D``, or ``nn.Module``
                compatible with kornia interface.

                If a sequence of params (``dict``) passed, then each
                element of the sequence must contain ``'transform'`` key with
                an augmentation name as a value. Please note that in this case
                to use custom augmentation you should add it to the
                `REGISTRY` registry first.
            input_key (Union[str, int]): key in batch dict
                mapping to transform, e.g. `'image'`
            output_key: key to use to store the result
                of the transform, defaults to `input_key` if not provided
        """
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.all)

        self.input_key = input_key
        self.output_key = output_key or self.input_key

        transforms: Sequence[nn.Module] = [
            item if isinstance(item, nn.Module) else REGISTRY.get_from_params(**item)
            for item in transform
        ]
        assert all(
            isinstance(t, nn.Module) for t in transforms
        ), "`nn.Module` should be a base class for transforms"

        self.transform = nn.Sequential(*transforms)

    def on_batch_start(self, runner: "IRunner") -> None:
        """Apply transforms.

        Args:
            runner: сurrent runner
        """
        input_batch = runner.input[self.input_key]
        output_batch = self.transform(input_batch)
        runner.input[self.output_key] = output_batch


__all__ = ["BatchTransformCallback"]
