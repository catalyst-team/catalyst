from collections import defaultdict
from typing import Any, Dict, Tuple, Iterable, Union
from abc import ABC, abstractmethod

import numpy as np
import torch

from catalyst.metrics import get_binary_statistics, get_multiclass_statistics
from catalyst.metrics.accuracy import accuracy
from catalyst.metrics.auc import auc
from catalyst.tools.meters.ppv_tpr_f1_meter import precision, recall, f1score


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
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
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

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
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
            self.mean = self.mean_old + (value - self.mean_old) * num_samples / float(
                self.num_samples
            )
            self.m_s += (value - self.mean_old) * (value - self.mean) * num_samples
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.num_samples - 1.0))
        return value

    def compute(self) -> Tuple[float, float]:
        return self.mean, self.std


class AccuracyMetric(ICallbackBatchMetric, AdditiveValueMetric):
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
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

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        value = self.update(logits=logits, targets=targets)
        return {self.metric_name_mean: value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {self.metric_name_mean: mean, self.metric_name_std: std}


class AUCMetric(ICallbackLoaderMetric):
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
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
            f"{self.metric_name}/class_{i+1:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[self.metric_name] = per_class_auc.mean().item()
        return output


class ConfusionMetric(IMetric):
    def __init__(self, num_classes: int, normalized: bool = False, compute_on_call: bool = True):
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
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class PrecisionRecallF1SupportMetric(ICallbackLoaderMetric):
    def __init__(
            self,
            compute_on_call: bool = True,
            prefix: str = None,
            suffix: str = None,
            mode: str = "binary",
            num_classes: int = 2,
            threshold: Union[float, Iterable[float]] = 0.5,
    ) -> None:
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        assert mode in ("binary", "micro", "macro", "weighted", None)
        self.mode = mode
        self.threshold = threshold
        self.num_classes = num_classes
        self.statistics = {}
        self.reset(num_batches=0, num_samples=0)

    def reset(self, num_batches, num_samples) -> None:
        if self.mode == "binary":
            self.statistics = defaultdict(float)
        else:
            self.statistics = defaultdict(lambda: torch.zeros(size=(self.num_classes, )))

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        # outputs = (scores > self.threshold).float()

        if self.mode == "binary":
            _, fp, fn, tp, support = get_binary_statistics(outputs=outputs, targets=targets)
        else:
            _, fp, fn, tp, support = get_multiclass_statistics(
                outputs=outputs, targets=targets, num_classes=self.num_classes
            )
        self.statistics["fp"] += fp
        self.statistics["fn"] += fn
        self.statistics["tp"] += tp
        self.statistics["support"] += support

    def compute(self) -> Any:
        if self.mode == "binary":
            precision_value = precision(tp=self.statistics["tp"], fp=self.statistics["fp"])
            recall_value = recall(tp=self.statistics["tp"], fn=self.statistics["fn"])
            f1_value = f1score(precision_value=precision_value, recall_value=recall_value)

        else:
            precision_values, recall_values, f1_values = \
                torch.zeros(size=(self.num_classes, )), \
                torch.zeros(size=(self.num_classes, )), \
                torch.zeros(size=(self.num_classes, ))
            for i in range(self.num_classes):
                precision_values[i] = precision(
                    tp=self.statistics["tp"][i], fp=self.statistics["fp"][i]
                )
                recall_values[i] = recall(tp=self.statistics["tp"][i], fn=self.statistics["fn"][i])
                f1_values[i] = f1score(
                    precision_value=precision_values[i], recall_value=recall_values[i]
                )
            precision_mask = (self.statistics["tp"] + self.statistics["fp"] > 0).bool()
            # if there is a label that not in targets, recall value is set to 1
            # sklearn metrics set it with 0
            recall_mask = (self.statistics["support"] > 0).bool()
            if self.mode == "macro":
                precision_value = (precision_values * precision_mask).float().mean()
                recall_value = (recall_values * recall_mask).float().mean()
                f1_value = f1_values.mean()

            elif self.mode == "weighted":
                weight = self.statistics["support"] / self.statistics["support"].sum()
                precision_value = (precision_values * precision_mask * weight).sum()
                recall_value = (recall_values * recall_mask * weight).sum()
                f1_value = (f1_values * weight).sum()

            elif self.mode == "micro":
                precision_value = self.statistics["tp"].sum() / (
                        self.statistics["tp"].sum() + self.statistics["fp"].sum()
                )
                recall_value = self.statistics["tp"].sum() / (
                        self.statistics["tp"].sum() + self.statistics["fn"].sum()
                )
                f1_value = 2 * self.statistics["tp"].sum() / (
                        2 * self.statistics["tp"].sum() +
                        self.statistics["fp"].sum() + self.statistics["fn"].sum()
                )
            else:
                return precision_values * precision_mask, recall_values * recall_mask, f1_values

        return precision_value, recall_value, f1_value

    def compute_key_value(self) -> Dict[str, float]:
        precision_value, recall_value, f1_value = self.compute()
        if self.mode is None:
            metrics = {}
            for i in range(self.num_classes):
                for (metric_name, metric_value) in zip(
                        ("precision", "recall", "f1"), (precision_value, recall_value, f1_value)
                ):
                    metrics[f"{metric_name}/class_{i + 1}"] = metric_value
            return metrics
        return {"precision": precision_value, "recall": recall_value, "f1": f1_value}
