from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._classification import get_aggregated_metrics, get_binary_metrics
from catalyst.metrics.functional._misc import (
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)
from catalyst.utils.distributed import all_gather, get_rank


class StatisticsMetric(ICallbackBatchMetric):
    """
    This metric accumulates true positive, false positive, true negative,
    false negative, support statistics from data.

    It can work in binary, multiclass and multilabel classification tasks.

    Args:
        mode: one of "binary", "multiclass" and "multilabel"
        num_classes: number of classes
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

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
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
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

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        mode: str,
        num_classes: int = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init params

        Raises:
            ValueError: if mode is incorrect
        """
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix,
        )
        if mode == "binary":
            self.statistics_fn = get_binary_statistics
        elif mode == "multiclass":
            self.statistics_fn = partial(get_multiclass_statistics, num_classes=num_classes)
        elif mode == "multilabel":
            self.statistics_fn = get_multilabel_statistics
        else:
            raise ValueError("Mode should be one of 'binary', 'multiclass', 'multilabel'")

        self.num_classes = num_classes
        self.statistics = None
        self._is_ddp = False
        self.reset()

    # multiprocessing could not handle lamdas, so..
    def _mp_hack(self):
        return np.zeros(shape=(self.num_classes,))

    def reset(self) -> None:
        """Reset all the statistics."""
        self.statistics = defaultdict(self._mp_hack)
        self._is_ddp = get_rank() > -1

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Tuple[int, int, int, int, int], Tuple[Any, Any, Any, Any, Any]]:
        """
        Compute statistics from outputs and targets, update accumulated statistics with new values.

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            Tuple of int or array: true negative, false positive, false
                negative, true positive and support statistics
        """
        tn, fp, fn, tp, support = self.statistics_fn(
            outputs=outputs.cpu().detach(), targets=targets.cpu().detach(),
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

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
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
            >>> For binary mode: {"tp": 3, "fp": 4, "tn": 5, "fn": 1, "support": 13}
            >>> For other modes: {"tp": np.array([1, 2, 1]), "fp": np.array([2, 1, 0]), ...}
        """
        result = self.compute()
        return {k: result[k] for k in sorted(result.keys())}


class PrecisionRecallF1SupportMetric(StatisticsMetric):
    """
    Metric that can collect statistics and count precision, recall, f1_score and support with it.

    Args:
        mode: one of "binary", "multiclass" and "multilabel"
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metrics prefix
        suffix: metrics suffix
    """

    def __init__(
        self,
        mode: str,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ) -> None:
        """Init PrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            mode=mode,
        )
        self.zero_division = zero_division
        self.reset()

    def _convert_metrics_to_kv(self, per_class, micro, macro, weighted) -> Dict[str, float]:
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

        per_class_metrics = {
            f"{metric_name}/class_{i:02d}": metric_value[i]
            for metric_name, metric_value in zip(
                ("precision", "recall", "f1", "support"), per_class
            )
            for i in range(self.num_classes)  # noqa: WPS361
        }
        kv_metrics.update(per_class_metrics)
        return kv_metrics

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            tuple of metrics intermediate results with per-class, micro, macro and
                weighted averaging
        """
        tn, fp, fn, tp, support = super().update(outputs=outputs, targets=targets)
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=tp, fp=fp, fn=fn, support=support, zero_division=self.zero_division,
        )
        return per_class, micro, macro, weighted

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
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
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            support=self.statistics["support"],
            zero_division=self.zero_division,
        )
        return per_class, micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics
        """
        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            for key in self.statistics:
                value: List[np.ndarray] = all_gather(self.statistics[key])
                value: np.ndarray = np.sum(np.vstack(value), axis=0)
                self.statistics[key] = value

        per_class, micro, macro, weighted = self.compute()
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics


class BinaryPrecisionRecallF1Metric(StatisticsMetric):
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
            num_classes=2,
            mode="binary",
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
        )
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
        self.statistics = defaultdict(float)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
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

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics and return metrics intermediate results

        Args:
            outputs: predicted labels
            targets: target labels

        Returns:
            dict of intermediate metrics
        """
        precision_value, recall_value, f1_value = self.update(outputs=outputs, targets=targets)
        kv_metrics = self._convert_metrics_to_kv(
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value,
        )
        return kv_metrics

    def compute(self) -> Tuple[float, float, float]:
        """
        Compute metrics with accumulated statistics

        Returns:
            tuple of metrics: precision, recall, f1 score
        """
        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            for key in self.statistics:
                value: List[float] = all_gather(self.statistics[key])
                value: float = sum(value)
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
            precision_value=precision_value, recall_value=recall_value, f1_value=f1_value,
        )
        return kv_metrics


class MulticlassPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multiclass classification.
    Counts metrics with macro, micro and weighted average.

    Args:
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        num_classes = 4
        zero_division = 0
        outputs_list = [torch.tensor([0, 1, 2]), torch.tensor([2, 3]), torch.tensor([0, 1, 3])]
        targets_list = [torch.tensor([0, 1, 1]), torch.tensor([2, 3]), torch.tensor([0, 1, 2])]

        metric = metrics.MulticlassPrecisionRecallF1SupportMetric(
            num_classes=num_classes, zero_division=zero_division
        )
        metric.reset()

        for outputs, targets in zip(outputs_list, targets_list):
            metric.update(outputs=outputs, targets=targets)

        metric.compute()
        # (
        #     # per class precision, recall, f1, support
        #     (
        #         array([1. , 1. , 0.5, 0.5]),
        #         array([1.        , 0.66666667, 0.5       , 1.        ]),
        #         array([0.999995  , 0.7999952 , 0.499995  , 0.66666222]),
        #         array([2., 3., 2., 1.]),
        #     ),
        #     # micro precision, recall, f1, support
        #     (0.75, 0.75, 0.7499950000333331, None),
        #     # macro precision, recall, f1, support
        #     (0.75, 0.7916666666666667, 0.7416618555889127, None),
        #     # weighted precision, recall, f1, support
        #     (0.8125, 0.75, 0.7583284778110313, None)
        # )

        metric.compute_key_value()
        # {
        #     'f1/_macro': 0.7416618555889127,
        #     'f1/_micro': 0.7499950000333331,
        #     'f1/_weighted': 0.7583284778110313,
        #     'f1/class_00': 0.9999950000249999,
        #     'f1/class_01': 0.7999952000287999,
        #     'f1/class_02': 0.49999500004999947,
        #     'f1/class_03': 0.6666622222518517,
        #     'precision/_macro': 0.75,
        #     'precision/_micro': 0.75,
        #     'precision/_weighted': 0.8125,
        #     'precision/class_00': 1.0,
        #     'precision/class_01': 1.0,
        #     'precision/class_02': 0.5,
        #     'precision/class_03': 0.5,
        #     'recall/_macro': 0.7916666666666667,
        #     'recall/_micro': 0.75,
        #     'recall/_weighted': 0.75,
        #     'recall/class_00': 1.0,
        #     'recall/class_01': 0.6666666666666667,
        #     'recall/class_02': 0.5,
        #     'recall/class_03': 1.0,
        #     'support/class_00': 2.0,
        #     'support/class_01': 3.0,
        #     'support/class_02': 2.0,
        #     'support/class_03': 1.0
        # }

        metric.reset()
        metric(outputs_list[0], targets_list[0])
        # (
        #     # per class precision, recall, f1, support
        #     (
        #         array([1., 1., 0., 0.]),
        #         array([1. , 0.5, 0. , 0. ]),
        #         array([0.999995  , 0.66666222, 0.        , 0.        ]),
        #         array([1., 2., 0., 0.]),
        #     ),
        #     # micro precision, recall, f1, support
        #     (0.6666666666666667, 0.6666666666666667, 0.6666616667041664, None),
        #     # macro precision, recall, f1, support
        #     (0.5, 0.375, 0.41666430556921286, None),
        #     # weighted precision, recall, f1, support
        #     (1.0, 0.6666666666666666, 0.7777731481762343, None)
        # )

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
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
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

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init MultiClassPrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            zero_division=zero_division,
            mode="multiclass",
        )


class MultilabelPrecisionRecallF1SupportMetric(PrecisionRecallF1SupportMetric):
    """
    Precision, recall, f1_score and support metrics for multilabel classification.
    Counts metrics with macro, micro and weighted average.

    Args:
        num_classes: number of classes in loader's dataset
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        num_classes = 4
        zero_division = 0
        outputs_list = [
            torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]]),
            torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]]),
            torch.tensor([[0, 1, 0, 0], [0, 1, 0, 1]]),
        ]
        targets_list = [
            torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]]),
            torch.tensor([[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]),
            torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]]),
        ]

        metric = metrics.MultilabelPrecisionRecallF1SupportMetric(
            num_classes=num_classes, zero_division=zero_division
        )
        metric.reset()

        for outputs, targets in zip(outputs_list, targets_list):
            metric.update(outputs=outputs, targets=targets)

        metric.compute()
        # (
        #     # per class precision, recall, f1, support
        #     (
        #         array([0.        , 0.66666667, 0.        , 0.4       ]),
        #         array([0.        , 1.        , 0.        , 0.66666667]),
        #         array([0.        , 0.7999952 , 0.        , 0.49999531]),
        #         array([1., 4., 4., 3.])
        #     ),
        #     # micro precision, recall, f1, support
        #     (0.46153846153846156, 0.5, 0.4799950080519163, None),
        #     # macro precision, recall, f1, support
        #     (0.2666666666666667, 0.4166666666666667, 0.32499762814318617, None),
        #     # weighted precision, recall, f1, support
        #     (0.32222222222222224, 0.5, 0.39166389481225283, None)
        # )

        metric.compute_key_value()
        # {
        #     'f1/_macro': 0.32499762814318617,
        #     'f1/_micro': 0.4799950080519163,
        #     'f1/_weighted': 0.39166389481225283,
        #     'f1/class_00': 0.0,
        #     'f1/class_01': 0.7999952000287999,
        #     'f1/class_02': 0.0,
        #     'f1/class_03': 0.49999531254394486,
        #     'precision/_macro': 0.2666666666666667,
        #     'precision/_micro': 0.46153846153846156,
        #     'precision/_weighted': 0.32222222222222224,
        #     'precision/class_00': 0.0,
        #     'precision/class_01': 0.6666666666666667,
        #     'precision/class_02': 0.0,
        #     'precision/class_03': 0.4,
        #     'recall/_macro': 0.4166666666666667,
        #     'recall/_micro': 0.5,
        #     'recall/_weighted': 0.5,
        #     'recall/class_00': 0.0,
        #     'recall/class_01': 1.0,
        #     'recall/class_02': 0.0,
        #     'recall/class_03': 0.6666666666666667,
        #     'support/class_00': 1.0,
        #     'support/class_01': 4.0,
        #     'support/class_02': 4.0,
        #     'support/class_03': 3.0
        # }


        metric.reset()
        metric(outputs_list[0], targets_list[0])
        # (
        #     # per class precision, recall, f1, support
        #     (
        #         array([0., 1., 0., 1.]),
        #         array([0. , 1. , 0. , 0.5]),
        #         array([0.        , 0.999995  , 0.        , 0.66666222]),
        #         array([0., 2., 1., 2.])
        #     ),
        #     # micro precision, recall, f1, support
        #     (0.75, 0.6, 0.6666617284316411, None),
        #     # macro precision, recall, f1, support
        #     (0.5, 0.375, 0.41666430556921286, None),
        #     # weighted precision, recall, f1, support
        #     (0.8, 0.6000000000000001, 0.6666628889107407, None)
        # )

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
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
            valid_metric="accuracy",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.BatchTransformCallback(
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.MultilabelAccuracyCallback(
                    input_key="scores", target_key="targets", threshold=0.5
                ),
                dl.MultilabelPrecisionRecallF1SupportCallback(
                    input_key="scores", target_key="targets", threshold=0.5
                ),
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        num_classes: int = None,
        zero_division: int = 0,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init MultiLabelPrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            num_classes=num_classes,
            zero_division=zero_division,
            mode="multilabel",
        )


__all__ = [
    "BinaryPrecisionRecallF1Metric",
    "MulticlassPrecisionRecallF1SupportMetric",
    "MultilabelPrecisionRecallF1SupportMetric",
]
