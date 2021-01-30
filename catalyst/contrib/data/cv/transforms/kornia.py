from typing import Any, Dict, Iterable, Optional, Tuple, Union
import copy
import random

from kornia.augmentation import AugmentationBase2D, AugmentationBase3D
import numpy as np
import torch
from torch import nn


class OneOfPerBatch(nn.Module):
    """Select one of tensor transforms and apply it batch-wise."""

    def __init__(
        self, transforms: Iterable[Union[AugmentationBase2D, AugmentationBase3D]],
    ) -> None:
        """Constructor method for the :class:`OneOfPerBatch` transform.

        Args:
            transforms: list of kornia transformations to compose.
                Actually, any ``nn.Module`` with defined ``p``(probability
                of selecting transform) and ``p_batch`` attributes is allowed.
        """
        super().__init__()

        probs = [transform.p for transform in transforms]
        s = sum(probs)
        self.probs = [proba / s for proba in probs]

        self.transforms = [copy.deepcopy(t) for t in transforms]
        for t in self.transforms:
            t.p = 1
            t.p_batch = 1

    def forward(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[Dict[str, torch.Tensor]] = None,
        return_transform: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply transform.

        Args:
            input: input batch
            params: transform params, please check kornia documentation
            return_transform: if ``True`` return the matrix describing
                the geometric transformation applied to each input tensor,
                please check kornia documentation

        Returns:
            augmented batch and, optionally, the transformation matrix
        """
        # select transform to apply
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        t = random_state.choice(self.transforms, p=self.probs)

        # apply kornia transform
        output = t(input, params, return_transform)

        return output


class OneOfPerSample(nn.Module):
    """Select one of tensor transforms to apply sample-wise."""

    def __init__(
        self, transforms: Iterable[Union[AugmentationBase2D, AugmentationBase3D]],
    ) -> None:
        """Constructor method for the :class:`OneOfPerSample` transform.

        Args:
            transforms: list of kornia transformations to compose.
                Actually, any ``nn.Module`` with defined ``p``(probability
                of selecting transform) and ``p_batch`` attributes is allowed.
        """
        super().__init__()

        probs = [transform.p for transform in transforms]
        s = sum(probs)
        self.choice_transform = torch.distributions.Categorical(
            torch.tensor([proba / s for proba in probs])
        )

        self.transforms = [copy.deepcopy(t) for t in transforms]
        for t in self.transforms:
            t.p = 1
            t.p_batch = 1

    def forward(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[Dict[str, torch.Tensor]] = None,
        return_transform: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply transform.

        Args:
            input: input batch
            params: transform params, please check kornia documentation
            return_transform: if ``True`` return the matrix describing
                the geometric transformation applied to each input tensor,
                please check kornia documentation

        Returns:
            augmented batch and, optionally, the transformation matrix
        """
        # select transform for each element
        batch_size = (input[0] if isinstance(input, tuple) else input).shape[0]
        transforms = self.choice_transform.sample([batch_size])

        # apply transforms
        for idx, transform in enumerate(self.transforms):
            to_apply = transforms == idx
            if to_apply.any():
                # TODO: return transform matrix if `return_transform` == True
                self._apply_transform(
                    transform,
                    batch=input,
                    mask=to_apply,
                    params=params,
                    return_transform=return_transform,
                )

        return input

    @staticmethod
    def _apply_transform(
        transform: nn.Module,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        *args: Any,
        return_transform: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Apply ``transform`` inplace."""
        # process input
        input_ = (batch[0][mask], batch[1][mask]) if isinstance(batch, tuple) else batch[mask]

        output = transform(input_, *args, transform_matrix=return_transform, **kwargs)

        # process output
        transform_matrix = None
        if return_transform:
            output, transform_matrix = output

        if isinstance(batch, tuple):
            batch[0][mask] = output[0]
            batch[1][mask] = output[1]

        return transform_matrix


__all__ = ["OneOfPerBatch", "OneOfPerSample"]
