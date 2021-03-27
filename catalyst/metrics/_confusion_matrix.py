from typing import Any, List

import numpy as np
import torch

from catalyst.metrics._metric import IMetric
from catalyst.utils.distributed import all_gather, get_rank


class ConfusionMatrixMetric(IMetric):
    """Constructs a confusion matrix for a multiclass classification problems.

    Args:
        num_classes: number of classes in the classification problem
        normalized: determines whether or not the confusion matrix is normalized or not
        compute_on_call: Boolean flag to computes and return confusion matrix during __call__.
            default: True
    """

    def __init__(self, num_classes: int, normalized: bool = False, compute_on_call: bool = True):
        """Constructs a confusion matrix for a multiclass classification problems."""
        super().__init__(compute_on_call=compute_on_call)
        self.num_classes = num_classes
        self.normalized = normalized
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self._is_ddp = False
        self.reset()

    def reset(self) -> None:
        """Reset confusion matrix, filling it with zeros."""
        self.conf.fill(0)
        self._is_ddp = get_rank() > -1

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Computes the confusion matrix of ``K x K`` size where ``K`` is no of classes.

        Args:
            predictions: Can be an N x K tensor of predicted scores
                obtained from the model for N examples and K classes
                or an N-tensor of integer values between 0 and K-1
            targets: Can be a N-tensor of integer values assumed
                to be integer values between 0 and K-1 or N x K tensor, where
                targets are assumed to be provided as one-hot vectors
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        assert (
            predictions.shape[0] == targets.shape[0]
        ), "number of targets and predicted outputs do not match"

        if np.ndim(predictions) != 1:
            assert (
                predictions.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predictions = np.argmax(predictions, 1)
        else:
            assert (predictions.max() < self.num_classes) and (
                predictions.min() >= 0
            ), "predicted values are not between 1 and k"

        onehot_target = np.ndim(targets) != 1
        if onehot_target:
            assert (
                targets.shape[1] == self.num_classes
            ), "Onehot target does not match size of confusion matrix"
            assert (targets >= 0).all() and (
                targets <= 1
            ).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (targets.sum(1) == 1).all(), "multilabel setting is not supported"
            targets = np.argmax(targets, 1)
        else:
            assert (predictions.max() < self.num_classes) and (
                predictions.min() >= 0
            ), "predicted values are not between 0 and k-1"

        # hack for bincounting 2 arrays together
        x = predictions + self.num_classes * targets
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes ** 2
        )  # noqa: WPS114
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def compute(self) -> Any:
        """
        Returns:
            Confusion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self._is_ddp:
            value: List[np.ndarray] = all_gather(self.conf)
            value: np.ndarray = np.sum(np.stack(value, axis=0), axis=0)
            self.conf = value
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


__all__ = ["ConfusionMatrixMetric"]
