from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from functools import partial

import numpy as np

import torch

from catalyst.metrics.accuracy import accuracy
from catalyst.metrics.auc import auc
from catalyst.metrics.region_base_metrics import (
    _dice,
    _iou,
    _trevsky,
    get_segmentation_statistics,
)


class IMetric(ABC):
    """Interface for all Metrics."""

    def __init__(self, compute_on_call: bool = True):
        """Interface for all Metrics.

        Args:
            compute_on_call:
                Computes and returns metric value during metric call.
                Used for per-batch logging. default: True
        """
        self.compute_on_call = compute_on_call

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Updates the metrics state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).

        Args:
            *args: some args :)
            **kwargs: some kwargs ;)
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Any: computed value, # noqa: DAR202
            it's better to return key-value
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each batch
        (`on_batch_end` event).
        Returns computed value if `compute_on_call=True`.

        Returns:
            Any: computed value, it's better to return key-value.
        """
        value = self.update(*args, **kwargs)
        return self.compute() if self.compute_on_call else value


class ICallbackBatchMetric(IMetric):
    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def update_key_value(self, *args, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        pass


class ICallbackLoaderMetric(IMetric):
    """Interface for all Metrics."""

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def reset(self, num_batches, num_samples) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metrics state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).

        Args:
            *args: some args :)
            **kwargs: some kwargs ;)
        """
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        # @TODO: could be refactored - we need custom exception here
        # we need this method only for callback metric logging
        pass


class AdditiveValueMetric(IMetric):
    def __init__(self, compute_on_call: bool = True):
        super().__init__(compute_on_call=compute_on_call)
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def reset(self) -> None:
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def update(self, value: float, num_samples: int) -> float:
        self.value = value
        self.n += 1
        self.num_samples += num_samples

        if self.n == 1:
            # Force a copy in torch/numpy
            self.mean = 0.0 + value  # noqa: WPS345
            self.std = 0.0
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (
                value - self.mean_old
            ) * num_samples / float(self.num_samples)
            self.m_s += (
                (value - self.mean_old) * (value - self.mean) * num_samples
            )
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.num_samples - 1.0))
        return value

    def compute(self) -> Tuple[float, float]:
        return self.mean, self.std


class AccuracyMetric(ICallbackBatchMetric, AdditiveValueMetric):
    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        ICallbackBatchMetric.__init__(
            self, compute_on_call=compute_on_call, prefix=prefix, suffix=suffix
        )
        AdditiveValueMetric.__init__(self, compute_on_call=compute_on_call)
        self.metric_name_mean = f"{self.prefix}accuracy{self.suffix}"
        self.metric_name_std = f"{self.prefix}accuracy{self.suffix}/std"

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        value = accuracy(logits, targets)[0].item()
        value = super().update(value, len(targets))
        return value

    def update_key_value(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        value = self.update(logits=logits, targets=targets)
        return {self.metric_name_mean: value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {self.metric_name_mean: mean, self.metric_name_std: std}


class AUCMetric(ICallbackLoaderMetric):
    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix
        )
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []

    def reset(self, num_batches, num_samples) -> None:
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> torch.Tensor:
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)
        score = auc(outputs=scores, targets=targets)
        return score

    def compute_key_value(self) -> Dict[str, float]:
        per_class_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i + 1:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[self.metric_name] = per_class_auc.mean().item()
        return output


class ConfusionMetric(IMetric):
    def __init__(
        self,
        num_classes: int,
        normalized: bool = False,
        compute_on_call: bool = True,
    ):
        """ConfusionMatrix constructs a confusion matrix for a multiclass classification problems.

        Args:
            num_classes: number of classes in the classification problem
            normalized: determines whether or not the confusion matrix is normalized or not
            compute_on_call:
        """
        super().__init__(compute_on_call=compute_on_call)
        self.num_classes = num_classes
        self.normalized = normalized
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.reset()

    def reset(self) -> None:
        """Reset confusion matrix, filling it with zeros."""
        self.conf.fill(0)

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
            assert (
                targets.sum(1) == 1
            ).all(), "multilabel setting is not supported"
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
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class RegionBasedMetric(ICallbackLoaderMetric):
    """
    Logic class for all region based metrics, like IoU, Dice, Trevsky
    """

    def __init__(
        self,
        metric_fn: Callable,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        """

        Args:
            metric_fn: metric function, that get statistics and return score
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
        """
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_fn = metric_fn
        self.class_dim = class_dim
        self.threshold = threshold
        self.statistics = {}
        self.weights = weights
        self.class_names = class_names
        self._checked_params = False

    def reset(self):
        self.statistics = {}

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        tp, fp, fn = get_segmentation_statistics(
            outputs=outputs.cpu().detach(),
            targets=targets.cpu().detach(),
            class_dim=self.class_dim,
            threshold=self.threshold,
        )

        for idx, (tp_class, fp_class, fn_class) in enumerate(
            zip(tp, fp, fn), start=1
        ):
            if idx in self.statistics:
                self.statistics[idx]["tp"] += tp_class
                self.statistics[idx]["fp"] += fp_class
                self.statistics[idx]["fn"] += fn_class
            else:
                self.statistics[idx] = {}
                self.statistics[idx]["tp"] = tp_class
                self.statistics[idx]["fp"] = fp_class
                self.statistics[idx]["fn"] = fn_class

        values = self.metric_fn(tp, fp, fn)
        return values

    def _check_parameters(self):
        # check class_names
        if self.class_names is not None:
            assert len(self.class_names) == len(self.statistics), (
                f"the number of"
                f" class names must"
                f" be equal to "
                f"the number of"
                f" classes, got"
                f" weights {len(self.class_names)}"
                f" and classes: {len(self.statistics)}"
            )
        else:
            self.class_names = [
                f"class_{idx}" for idx in range(1, len(self.statistics) + 1)
            ]
        if self.weights is not None:
            assert len(self.weights) == len(self.statistics), (
                f"the number of"
                f" weights must"
                f" be equal to "
                f"the number of"
                f" classes, got"
                f" weights {len(self.weights)}"
                f" and classes: {len(self.statistics)}"
            )

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        values = self.update(outputs, targets)
        # need only one time
        if not self._checked_params:
            self._check_parameters()
            self._checked_params = True
        metrics = {
            f"{self.prefix}/{self.class_names[idx-1]}": value
            for idx, value in enumerate(values, start=1)
        }
        if self.weights is not None:
            weighted_metric = 0
            for idx, value in enumerate(values):
                weighted_metric += value * self.weights[idx]
            metrics[f"{self.prefix}/weighted"] = weighted_metric
        return metrics

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        metrics = {}
        total_statistics = {}
        micro_metric = 0
        if self.weights is not None:
            weighted_metric = 0
        for class_idx, statistics in self.statistics.items():
            value = self.metric_fn(**statistics)
            micro_metric += value
            if self.weights is not None:
                weighted_metric += value * self.weights[class_idx - 1]
            metrics[f"{self.prefix}/{self.class_names[class_idx-1]}"] = value
            for stats_name, value in statistics.items():
                total_statistics[stats_name] = (
                    total_statistics.get(stats_name, 0) + value
                )
        micro_metric /= len(self.statistics)
        macro_metric = self.metric_fn(**total_statistics)
        metrics[f"{self.prefix}/micro"] = micro_metric
        metrics[f"{self.prefix}/macro"] = macro_metric
        if self.weights is not None:
            metrics[f"{self.prefix}/weighted"] = weighted_metric
        return metrics

    def compute(self):
        return self.compute_key_value()


class IOUMetric(RegionBasedMetric):
    """
    IoU Metric
    iou score = intersection / union = tp / (tp + fp + fn)
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        metric_fn = partial(_iou, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class DiceMetric(RegionBasedMetric):
    """
    Dice Metric
    dice score = 2 * intersection / (intersection + union)) = \
    = 2 * tp / (2 * tp + fp + fn)
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        metric_fn = partial(_dice, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class TrevskyMetric(RegionBasedMetric):
    """
    Trevsky Metric
    trevsky score = tp / (tp + fp * beta + fn * alpha)
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. if beta is None, alpha must be in (0, 1)
            beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        if beta is None:
            assert 0 < alpha < 1, "if beta=None, alpha must be in (0, 1)"
            beta = 1 - alpha
        metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )
