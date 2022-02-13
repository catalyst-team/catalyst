from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np

import torch

from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._classification import (
    get_aggregated_metrics,
    get_binary_metrics,
)
from catalyst.metrics.functional._misc import (
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)
from catalyst.settings import SETTINGS
from catalyst.utils import get_device
from catalyst.utils.distributed import all_gather, get_backend

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


class BinaryStatisticsMetric(ICallbackBatchMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from binary data.

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

    Raises:
        ValueError: if mode is incorrect

    Examples:
    .. code-block:: python
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl
        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)
        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}
        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        # model training
        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
        )
    .. note::
        Please follow the `minimal examples`_ sections for more use cases.
        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init params"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.statistics = None
        self.num_classes = 2
        self._ddp_backend = None
        self.reset()

    # multiprocessing could not handle lamdas, so..
    def _mp_hack(self):
        return np.zeros(shape=(self.num_classes,))

    def reset(self) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(self._mp_hack)
        self._ddp_backend = get_backend()

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Tuple[int, int, int, int, int], Tuple[Any, Any, Any, Any, Any]]:
        """
        Compute statistics from outputs and targets,
        update accumulated statistics with new values.

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            Tuple of int or array: true negative, false positive, false
                negative, true positive and support statistics

        """
        tn, fp, fn, tp, support = get_binary_statistics(
            outputs=outputs.cpu().detach(), targets=targets.cpu().detach()
        )

        tn = tn.numpy()
        fp = fp.numpy()
        fn = fn.numpy()
        tp = tp.numpy()
        support = support.numpy()

        self.statistics["tn"] += tn
        self.statistics["fp"] += fp
        self.statistics["fn"] += fn
        self.statistics["tp"] += tp
        self.statistics["support"] += support

        return tn, fp, fn, tp, support

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return statistics intermediate result

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of statistics for current input

        """
        tn, fp, fn, tp, support = self.update(outputs=outputs, targets=targets)
        return {"fn": fn, "fp": fp, "support": support, "tn": tn, "tp": tp}

    def compute(self) -> Dict[str, Union[int, np.array]]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        """
        return self.statistics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        Examples:
            >>> {"tp": 3, "fp": 4, "tn": 5, "fn": 1, "support": 13}

        """
        result = self.compute()
        return {k: result[k] for k in sorted(result.keys())}


class MulticlassStatisticsMetric(ICallbackBatchMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from multiclass data.

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
        num_classes: number of classes

    Raises:
        ValueError: if mode is incorrect

    Examples:
    .. code-block:: python
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl
        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)
        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}
        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        # model training
        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
        )
    .. note::
        Please follow the `minimal examples`_ sections for more use cases.
        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """Init params"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.statistics = None
        self.num_classes = num_classes
        self._ddp_backend = None
        self.reset()

    # multiprocessing could not handle lamdas, so..
    def _mp_hack(self):
        return np.zeros(shape=(self.num_classes,))

    def reset(self) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(self._mp_hack)
        self._ddp_backend = get_backend()

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Tuple[int, int, int, int, int, int], Tuple[Any, Any, Any, Any, Any, int]]:
        """
        Compute statistics from outputs and targets,
        update accumulated statistics with new values.

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            Tuple of int or array: true negative, false positive, false
                negative, true positive, support statistics and num_classes

        """
        tn, fp, fn, tp, support, num_classes = get_multiclass_statistics(
            outputs=outputs.cpu().detach(),
            targets=targets.cpu().detach(),
            num_classes=self.num_classes,
        )

        tn = tn.numpy()
        fp = fp.numpy()
        fn = fn.numpy()
        tp = tp.numpy()
        support = support.numpy()

        if self.num_classes is None:
            self.num_classes = num_classes

        self.statistics["tn"] += tn
        self.statistics["fp"] += fp
        self.statistics["fn"] += fn
        self.statistics["tp"] += tp
        self.statistics["support"] += support

        return tn, fp, fn, tp, support, self.num_classes

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return statistics intermediate result

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of statistics for current input

        """
        tn, fp, fn, tp, support, _ = self.update(outputs=outputs, targets=targets)
        return {"fn": fn, "fp": fp, "support": support, "tn": tn, "tp": tp}

    def compute(self) -> Dict[str, Union[int, np.array]]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        """
        return self.statistics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        Examples:
            >>> {"tp": np.array([1, 2, 1]), "fp": np.array([2, 1, 0]), ...}

        """
        result = self.compute()
        return {k: result[k] for k in sorted(result.keys())}


class MultilabelStatisticsMetric(ICallbackBatchMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from multilabel data.

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
        num_classes: number of classes

    Raises:
        ValueError: if mode is incorrect

    Examples:
    .. code-block:: python
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl
        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)
        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}
        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        # model training
        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
        )
    .. note::
        Please follow the `minimal examples`_ sections for more use cases.
        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """Init params"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.statistics = None
        self.num_classes = num_classes
        self._ddp_backend = None
        self.reset()

    # multiprocessing could not handle lamdas, so..
    def _mp_hack(self):
        return np.zeros(shape=(self.num_classes,))

    def reset(self) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(self._mp_hack)
        self._ddp_backend = get_backend()

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Tuple[int, int, int, int, int, int], Tuple[Any, Any, Any, Any, Any, int]]:
        """
        Compute statistics from outputs and targets,
        update accumulated statistics with new values.

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            Tuple of int or array: true negative, false positive, false
                negative, true positive, support statistics and num_classes

        """
        tn, fp, fn, tp, support, num_classes = get_multilabel_statistics(
            outputs=outputs.cpu().detach(), targets=targets.cpu().detach()
        )

        tn = tn.numpy()
        fp = fp.numpy()
        fn = fn.numpy()
        tp = tp.numpy()
        support = support.numpy()
        if self.num_classes is None:
            self.num_classes = num_classes

        self.statistics["tn"] += tn
        self.statistics["fp"] += fp
        self.statistics["fn"] += fn
        self.statistics["tp"] += tp
        self.statistics["support"] += support

        return tn, fp, fn, tp, support, self.num_classes

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return statistics intermediate result

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of statistics for current input

        """
        tn, fp, fn, tp, support, _ = self.update(outputs=outputs, targets=targets)
        return {"fn": fn, "fp": fp, "support": support, "tn": tn, "tp": tp}

    def compute(self) -> Dict[str, Union[int, np.array]]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        """
        return self.statistics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Return accumulated statistics

        Returns:
            dict of statistics

        Examples:
            >>> {"tp": np.array([1, 2, 1]), "fp": np.array([2, 1, 0]), ...}

        """
        result = self.compute()
        return {k: result[k] for k in sorted(result.keys())}


class BinaryPrecisionRecallF1Metric(BinaryStatisticsMetric):
    """Precision, recall, f1_score and support metrics for binary classification.

    Args:
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix

    """

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init BinaryPrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
        )
        self.statistics = None
        self.zero_division = zero_division
        self.reset()

    @staticmethod
    def _convert_metrics_to_kv(
        precision_value: float, recall_value: float, f1_value: float
    ) -> Dict[str, float]:
        """
        Convert list of metrics to key-value

        Args:
            precision_value: precision value
            recall_value: recall value
            f1_value: f1 value

        Returns:
            dict of metrics

        """
        kv_metrics = {
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
        }
        return kv_metrics

    def reset(self) -> None:
        """Reset all the statistics and metrics fields."""
        self.statistics = defaultdict(int)

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Update statistics and return metrics intermediate results

        Args:
            outputs: predicted labels
            targets: target labels

        Returns:
            tuple of intermediate metrics: precision, recall, f1 score

        """
        tn, fp, fn, tp, support = super().update(outputs=outputs, targets=targets)
        precision_value, recall_value, f1_value = get_binary_metrics(
            tp=tp, fp=fp, fn=fn, zero_division=self.zero_division
        )
        return precision_value, recall_value, f1_value

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return metrics intermediate results

        Args:
            outputs: predicted labels
            targets: target labels

        Returns:
            dict of intermediate metrics

        """
        precision_value, recall_value, f1_value = self.update(
            outputs=outputs, targets=targets
        )
        kv_metrics = self._convert_metrics_to_kv(
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value
        )
        return kv_metrics

    def compute(self) -> Tuple[float, float, float]:
        """
        Compute metrics with accumulated statistics

        Returns:
            tuple of metrics: precision, recall, f1 score

        """
        # ddp hotfix, could be done better
        # but metric must handle DDP on it's own
        if self._ddp_backend == "xla":
            self.statistics = {
                k: xm.mesh_reduce(k, v, np.sum) for k, v in self.statistics.items()
            }
        elif self._ddp_backend == "ddp":
            for key in self.statistics:
                value: List[int] = all_gather(self.statistics[key])
                value: int = sum(value)
                self.statistics[key] = value

        precision_value, recall_value, f1_value = get_binary_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            zero_division=self.zero_division,
        )
        return precision_value, recall_value, f1_value

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute metrics with all accumulated statistics

        Returns:
            dict of metrics

        """
        precision_value, recall_value, f1_value = self.compute()
        kv_metrics = self._convert_metrics_to_kv(
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value
        )
        return kv_metrics


class MulticlassPrecisionRecallF1SupportMetric(MulticlassStatisticsMetric):
    """
    Metric that can collect statistics and count precision,
    recall, f1_score and support with it.

    Args:
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metrics prefix
        suffix: metrics suffix
        num_classes: number of classes

    """

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """Init PrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
        )
        self.compute_per_class_metrics = compute_per_class_metrics
        self.zero_division = zero_division
        self.num_classes = num_classes
        self.reset()

    def _convert_metrics_to_kv(
        self, per_class, micro, macro, weighted
    ) -> Dict[str, float]:
        """
        Convert metrics aggregation to key-value format

        Args:
            per_class: per-class metrics, array of shape (4, self.num_classes)
                of precision, recall, f1 and support metrics
            micro: micro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            macro: macro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            weighted: weighted averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics

        Returns:
            dict of key-value metrics

        """
        kv_metrics = {}
        for aggregation_name, aggregated_metrics in zip(
            ("_micro", "_macro", "_weighted"), (micro, macro, weighted)
        ):
            metrics = {
                f"{metric_name}/{aggregation_name}": metric_value
                for metric_name, metric_value in zip(
                    ("precision", "recall", "f1"), aggregated_metrics[:-1]
                )
            }
            kv_metrics.update(metrics)

        # @TODO: rewrite this block - should be without `num_classes`
        if self.compute_per_class_metrics:
            per_class_metrics = {
                f"{metric_name}/class_{i:02d}": metric_value[i]
                for metric_name, metric_value in zip(
                    ("precision", "recall", "f1", "support"), per_class
                )
                for i in range(self.num_classes)
            }
            kv_metrics.update(per_class_metrics)
        return kv_metrics

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            tuple of metrics intermediate results with per-class, micro, macro and
                weighted averaging

        """
        tn, fp, fn, tp, support, num_classes = super().update(
            outputs=outputs, targets=targets
        )
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=tp, fp=fp, fn=fn, support=support, zero_division=self.zero_division
        )
        if self.num_classes is None:
            self.num_classes = num_classes

        return per_class, micro, macro, weighted

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of metrics intermediate results

        """
        per_class, micro, macro, weighted = self.update(outputs=outputs, targets=targets)
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            list of aggregated metrics: per-class, micro, macro and weighted averaging of
                precision, recall, f1 score and support metrics

        """
        # ddp hotfix, could be done better
        # but metric must handle DDP on it's own
        if self._ddp_backend == "xla":
            device = get_device()
            for key in self.statistics:
                key_statistics = torch.tensor([self.statistics[key]], device=device)
                key_statistics = xm.all_gather(key_statistics).sum(dim=0).cpu().numpy()
                self.statistics[key] = key_statistics
        elif self._ddp_backend == "ddp":
            for key in self.statistics:
                value: List[np.ndarray] = all_gather(self.statistics[key])
                value: np.ndarray = np.sum(np.vstack(value), axis=0)
                self.statistics[key] = value

        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            support=self.statistics["support"],
            zero_division=self.zero_division,
        )
        if self.compute_per_class_metrics:
            return per_class, micro, macro, weighted
        else:
            return [], micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics

        """
        per_class, micro, macro, weighted = self.compute()
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics


class MultilabelPrecisionRecallF1SupportMetric(MultilabelStatisticsMetric):
    """
    Metric that can collect statistics and count precision,
    recall, f1_score and support with it.

    Args:
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metrics prefix
        suffix: metrics suffix
        num_classes: number of classes

    """

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """Init PrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
        )
        self.compute_per_class_metrics = compute_per_class_metrics
        self.zero_division = zero_division
        self.num_classes = num_classes
        self.reset()

    def _convert_metrics_to_kv(
        self, per_class, micro, macro, weighted
    ) -> Dict[str, float]:
        """
        Convert metrics aggregation to key-value format

        Args:
            per_class: per-class metrics, array of shape (4, self.num_classes)
                of precision, recall, f1 and support metrics
            micro: micro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            macro: macro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            weighted: weighted averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics

        Returns:
            dict of key-value metrics

        """
        kv_metrics = {}
        for aggregation_name, aggregated_metrics in zip(
            ("_micro", "_macro", "_weighted"), (micro, macro, weighted)
        ):
            metrics = {
                f"{metric_name}/{aggregation_name}": metric_value
                for metric_name, metric_value in zip(
                    ("precision", "recall", "f1"), aggregated_metrics[:-1]
                )
            }
            kv_metrics.update(metrics)

        # @TODO: rewrite this block - should be without `num_classes`
        if self.compute_per_class_metrics:
            per_class_metrics = {
                f"{metric_name}/class_{i:02d}": metric_value[i]
                for metric_name, metric_value in zip(
                    ("precision", "recall", "f1", "support"), per_class
                )
                for i in range(self.num_classes)
            }
            kv_metrics.update(per_class_metrics)
        return kv_metrics

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            tuple of metrics intermediate results with per-class, micro, macro and
                weighted averaging

        """
        tn, fp, fn, tp, support, num_classes = super().update(
            outputs=outputs, targets=targets
        )
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=tp, fp=fp, fn=fn, support=support, zero_division=self.zero_division
        )
        if self.num_classes is None:
            self.num_classes = num_classes

        return per_class, micro, macro, weighted

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of metrics intermediate results

        """
        per_class, micro, macro, weighted = self.update(outputs=outputs, targets=targets)
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            list of aggregated metrics: per-class, micro, macro and weighted averaging of
                precision, recall, f1 score and support metrics

        """
        # ddp hotfix, could be done better
        # but metric must handle DDP on it's own
        if self._ddp_backend == "xla":
            device = get_device()
            for key in self.statistics:
                key_statistics = torch.tensor([self.statistics[key]], device=device)
                key_statistics = xm.all_gather(key_statistics).sum(dim=0).cpu().numpy()
                self.statistics[key] = key_statistics
        elif self._ddp_backend == "ddp":
            for key in self.statistics:
                value: List[np.ndarray] = all_gather(self.statistics[key])
                value: np.ndarray = np.sum(np.vstack(value), axis=0)
                self.statistics[key] = value

        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            support=self.statistics["support"],
            zero_division=self.zero_division,
        )
        if self.compute_per_class_metrics:
            return per_class, micro, macro, weighted
        else:
            return [], micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics

        """
        per_class, micro, macro, weighted = self.compute()
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics


__all__ = [
    "BinaryPrecisionRecallF1Metric",
    "MulticlassPrecisionRecallF1SupportMetric",
    "MultilabelPrecisionRecallF1SupportMetric",
]
