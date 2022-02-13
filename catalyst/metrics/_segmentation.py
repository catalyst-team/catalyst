from typing import Callable, Dict, List, Optional
from functools import partial

import torch

from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._segmentation import (
    _dice,
    _iou,
    _trevsky,
    get_segmentation_statistics,
)
from catalyst.settings import SETTINGS
from catalyst.utils import get_device
from catalyst.utils.distributed import all_gather, get_backend

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


class RegionBasedMetric(ICallbackBatchMetric):
    """Logic class for all region based metrics, like IoU, Dice, Trevsky.

    Args:
        metric_fn: metric function, that get statistics and return score
        metric_name: name of the metric
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metric prefix
        suffix: metric suffix

    Interface, please check out implementations for more details:

        - :py:mod:`catalyst.metrics._segmentation.IOUMetric`
        - :py:mod:`catalyst.metrics._segmentation.DiceMetric`
        - :py:mod:`catalyst.metrics._segmentation.TrevskyMetric`
    """

    def __init__(
        self,
        metric_fn: Callable,
        metric_name: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = 0.5,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init"""
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.class_dim = class_dim
        self.threshold = threshold
        self.compute_per_class_metrics = compute_per_class_metrics
        # statistics = {class_idx: {"tp":, "fn": , "fp": "tn": }}
        self.statistics = {}
        self.weights = weights
        self.class_names = class_names
        self._checked_params = False
        self._ddp_backend = None

    def _check_parameters(self):
        # check class_names
        if self.class_names is not None:
            assert len(self.class_names) == len(self.statistics), (
                f"the number of class names must be equal to the number of classes,"
                " got weights"
                f" {len(self.class_names)} and classes: {len(self.statistics)}"
            )
        else:
            self.class_names = [
                f"class_{idx:02d}" for idx in range(len(self.statistics))
            ]
        if self.weights is not None:
            assert len(self.weights) == len(self.statistics), (
                f"the number of weights must be equal to the number of classes,"
                " got weights"
                f" {len(self.weights)} and classes: {len(self.statistics)}"
            )

    def reset(self):
        """Reset all statistics"""
        self.statistics = {}
        self._ddp_backend = get_backend()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Updates segmentation statistics with new data
        and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            metric for each class
        """
        tp, fp, fn = get_segmentation_statistics(
            outputs=outputs.cpu().detach(),
            targets=targets.cpu().detach(),
            class_dim=self.class_dim,
            threshold=self.threshold,
        )

        for idx, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
            if idx in self.statistics:
                self.statistics[idx]["tp"] += tp_class
                self.statistics[idx]["fp"] += fp_class
                self.statistics[idx]["fn"] += fn_class
            else:
                self.statistics[idx] = {}
                self.statistics[idx]["tp"] = tp_class
                self.statistics[idx]["fp"] = fp_class
                self.statistics[idx]["fn"] = fn_class

        # need only one time
        if not self._checked_params:
            self._check_parameters()
            self._checked_params = True

        metrics_per_class = self.metric_fn(tp, fp, fn)
        return metrics_per_class

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Updates segmentation statistics with new data
        and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            dict of metric for each class and weighted (if weights were given) metric
        """
        metrics_per_class = self.update(outputs, targets)
        macro_metric = torch.mean(metrics_per_class)
        metrics = {
            f"{self.prefix}{self.metric_name}{self.suffix}/{self.class_names[idx]}": val
            for idx, val in enumerate(metrics_per_class)
        }
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
        if self.weights is not None:
            weighted_metric = 0
            for idx, value in enumerate(metrics_per_class):
                weighted_metric += value * self.weights[idx]
            metrics[
                f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"
            ] = weighted_metric
        return metrics

    def compute(self):
        """
        Compute metrics with accumulated statistics

        Returns:
            tuple of metrics: per_class, micro_metric, macro_metric,
                weighted_metric(None if weights is None)
        """
        per_class = []
        total_statistics = {}
        macro_metric = 0
        weighted_metric = 0
        # ddp hotfix, could be done better
        # but metric must handle DDP on it's own
        # TODO: optimise speed
        if self._ddp_backend == "xla":
            device = get_device()
            for _, statistics in self.statistics.items():
                for key in statistics:
                    value = torch.tensor([statistics[key]], device=device)
                    statistics[key] = xm.all_gather(value).sum(dim=0)
        elif self._ddp_backend == "ddp":
            for _, statistics in self.statistics.items():
                for key in statistics:
                    value: List[torch.Tensor] = all_gather(statistics[key])
                    value: torch.Tensor = torch.sum(torch.vstack(value), dim=0)
                    statistics[key] = value

        for class_idx, statistics in self.statistics.items():
            value = self.metric_fn(**statistics)
            per_class.append(value)
            macro_metric += value
            if self.weights is not None:
                weighted_metric += value * self.weights[class_idx]
            for stats_name, value in statistics.items():
                total_statistics[stats_name] = (
                    total_statistics.get(stats_name, 0) + value
                )

        macro_metric /= len(self.statistics)
        micro_metric = self.metric_fn(**total_statistics)

        if self.weights is None:
            weighted_metric = None
        if self.compute_per_class_metrics:
            return per_class, micro_metric, macro_metric, weighted_metric
        else:
            return [], micro_metric, macro_metric, weighted_metric

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation metric for all data and return results in key-value format

        Returns:
             dict of metrics, including micro, macro
                and weighted (if weights were given) metrics
        """
        per_class, micro_metric, macro_metric, weighted_metric = self.compute()

        metrics = {}
        for class_idx, value in enumerate(per_class):
            class_name = self.class_names[class_idx]
            metrics[f"{self.prefix}{self.metric_name}{self.suffix}/{class_name}"] = value

        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_micro"] = micro_metric
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_macro"] = macro_metric
        if self.weights is not None:
            # @TODO: rename this one
            metrics[
                f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"
            ] = weighted_metric
        return metrics


class IOUMetric(RegionBasedMetric):
    """
    IoU Metric,
    iou score = intersection / union = tp / (tp + fp + fn).

    Args:
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        outputs = torch.tensor([[[[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]]]])
        targets = torch.tensor([[[[1.0, 0, 0], [0, 1, 0], [1, 1, 0]]]])
        metric = metrics.IOUMetric()
        metric.reset()

        metric.compute()
        # per_class, micro, macro, weighted
        # ([tensor(0.2222)], tensor(0.2222), tensor(0.2222), None)

        metric.update_key_value(outputs, targets)
        metric.compute_key_value()
        # {
        #     'iou': tensor(0.2222),
        #     'iou/_macro': tensor(0.2222),
        #     'iou/_micro': tensor(0.2222),
        #     'iou/class_00': tensor(0.2222),
        # }

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.contrib import IoULoss, MNIST

        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {
                    self._input_key: x, self._output_key: x_, self._target_key: x
                }

        runner = CustomRunner(
            input_key="features",
            output_key="scores",
            target_key="targets",
            loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        metric_fn = partial(_iou, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="iou",
            compute_on_call=compute_on_call,
            compute_per_class_metrics=compute_per_class_metrics,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class DiceMetric(RegionBasedMetric):
    """
    Dice Metric,
    dice score = 2 * intersection / (intersection + union)) = 2 * tp / (2 * tp + fp + fn)

    Args:
        class_dim: indicates class dimention (K) for ``outputs`` and
        ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        outputs = torch.tensor([[[[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]]]])
        targets = torch.tensor([[[[1.0, 0, 0], [0, 1, 0], [1, 1, 0]]]])
        metric = metrics.DiceMetric()
        metric.reset()

        metric.compute()
        # per_class, micro, macro, weighted
        # ([tensor(0.3636)], tensor(0.3636), tensor(0.3636), None)

        metric.update_key_value(outputs, targets)
        metric.compute_key_value()
        # {
        #     'dice': tensor(0.3636),
        #     'dice/_macro': tensor(0.3636),
        #     'dice/_micro': tensor(0.3636),
        #     'dice/class_00': tensor(0.3636),
        # }

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.contrib import IoULoss, MNIST

        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {
                    self._input_key: x, self._output_key: x_, self._target_key: x
                }

        runner = CustomRunner(
            input_key="features",
            output_key="scores",
            target_key="targets",
            loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        metric_fn = partial(_dice, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="dice",
            compute_on_call=compute_on_call,
            compute_per_class_metrics=compute_per_class_metrics,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class TrevskyMetric(RegionBasedMetric):
    """
    Trevsky Metric,
    trevsky score = tp / (tp + fp * beta + fn * alpha)

    Args:
        alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. if beta is None, alpha must be in (0, 1)
        beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        outputs = torch.tensor([[[[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]]]])
        targets = torch.tensor([[[[1.0, 0, 0], [0, 1, 0], [1, 1, 0]]]])
        metric = metrics.TrevskyMetric(alpha=0.2)
        metric.reset()

        metric.compute()
        # per_class, micro, macro, weighted
        # ([tensor(0.4167)], tensor(0.4167), tensor(0.4167), None)

        metric.update_key_value(outputs, targets)
        metric.compute_key_value()
        # {
        #     'trevsky': tensor(0.4167),
        #     'trevsky/_macro': tensor(0.4167)
        #     'trevsky/_micro': tensor(0.4167),
        #     'trevsky/class_00': tensor(0.4167),
        # }

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.contrib import IoULoss, MNIST

        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {
                    self._input_key: x, self._output_key: x_, self._target_key: x
                }

        runner = CustomRunner(
            input_key="features",
            output_key="scores",
            target_key="targets",
            loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        if beta is None:
            assert 0 < alpha < 1, "if beta=None, alpha must be in (0, 1)"
            beta = 1 - alpha
        metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="trevsky",
            compute_on_call=compute_on_call,
            compute_per_class_metrics=compute_per_class_metrics,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


__all__ = [
    "RegionBasedMetric",
    "IOUMetric",
    "DiceMetric",
    "TrevskyMetric",
]
