from typing import Any, Dict, Iterable, Optional, Tuple, Union
from collections import defaultdict

import numpy as np

import torch

from catalyst.metrics import (
    get_binary_statistics,
    get_multiclass_statistics,
    ICallbackLoaderMetric,
)
from catalyst.tools.meters.ppv_tpr_f1_meter import f1score, precision, recall


def precision_recall_fbeta_support(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-6,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Counts precision, recall, fbeta_score.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        beta: beta param for f_score
        eps: epsilon to avoid zero division
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known.

    Returns:
        tuple of precision, recall, fbeta_score

    Examples:
        >>> precision_recall_fbeta_support(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>>     beta=1,
        >>> )
        (
            tensor([1., 1., 1.]),  # precision per class
            tensor([1., 1., 1.]),  # recall per class
            tensor([1., 1., 1.]),  # fbeta per class
            tensor([1., 1., 1.]),  # support per class
        )
        >>> precision_recall_fbeta_support(
        >>>     outputs=torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),
        >>>     targets=torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),
        >>>     beta=1,
        >>> )
        (
            tensor([0.5000, 0.5000]),  # precision per class
            tensor([0.5000, 0.5000]),  # recall per class
            tensor([0.5000, 0.5000]),  # fbeta per class
            tensor([4., 4.]),          # support per class
        )
    """
    tn, fp, fn, tp, support = get_multiclass_statistics(
        outputs=outputs,
        targets=targets,
        argmax_dim=argmax_dim,
        num_classes=num_classes,
    )
    precision = (tp + eps) / (fp + tp + eps)
    recall = (tp + eps) / (fn + tp + eps)
    numerator = (1 + beta ** 2) * precision * recall
    denominator = beta ** 2 * precision + recall
    fbeta = numerator / denominator

    return precision, recall, fbeta, support


class PrecisionRecallF1SupportMetric(ICallbackLoaderMetric):
    """

    Notes:
        All the metrics (but support) for classes without true samples will be set with 1.
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        num_classes: int = 2,
        threshold: Union[float, Iterable[float]] = 0.5,
    ) -> None:
        """
        Init PrecisionRecallF1SupportMetric instance

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: ?
            suffix: ?
            num_classes: number of classes in loader's dataset
            threshold:
        """
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix
        )
        self.threshold = threshold
        self.num_classes = num_classes
        self.statistics = None
        self.metrics = None
        self.reset(batch_len=0, sample_len=0)

    def reset(self, batch_len: int, sample_len: int) -> None:
        """
        Reset all the statistics and metrics fields

        Args:
            batch_len: ?
            sample_len: ?
        """
        if self.num_classes == 2:
            self.statistics = defaultdict(float)
        else:
            self.statistics = defaultdict(
                lambda: np.zeros(shape=(self.num_classes,))
            )
        self.metrics = defaultdict(float)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update statistics with data from batch.

        Args:
            outputs: predicted labels
            targets: true labels
        """
        if self.num_classes == 2:
            _, fp, fn, tp, support = get_binary_statistics(
                outputs=outputs.cpu().detach(), targets=targets.cpu().detach()
            )
        else:
            _, fp, fn, tp, support = get_multiclass_statistics(
                outputs=outputs.cpu().detach(),
                targets=targets.cpu().detach(),
                num_classes=self.num_classes,
            )
        self.statistics["fp"] += fp.numpy()
        self.statistics["fn"] += fn.numpy()
        self.statistics["tp"] += tp.numpy()
        self.statistics["support"] += support.numpy()

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        If not binary, compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics
        """
        # binary mode
        if self.num_classes == 2:
            self.metrics["precision"] = precision(
                tp=self.statistics["tp"], fp=self.statistics["fp"]
            )
            self.metrics["recall"] = recall(
                tp=self.statistics["tp"], fn=self.statistics["fn"]
            )
            self.metrics["f1"] = f1score(
                precision_value=self.metrics["precision"],
                recall_value=self.metrics["recall"],
            )
        else:
            precision_values, recall_values, f1_values = (
                np.zeros(shape=(self.num_classes,)),
                np.zeros(shape=(self.num_classes,)),
                np.zeros(shape=(self.num_classes,)),
            )

            for i in range(self.num_classes):
                precision_values[i] = precision(
                    tp=self.statistics["tp"][i], fp=self.statistics["fp"][i]
                )
                recall_values[i] = recall(
                    tp=self.statistics["tp"][i], fn=self.statistics["fn"][i]
                )
                f1_values[i] = f1score(
                    precision_value=precision_values[i],
                    recall_value=recall_values[i],
                )

            weights = (
                self.statistics["support"] / self.statistics["support"].sum()
            )

            for metric_name, metric_value in zip(
                ("precision", "recall", "f1", "support"),
                (
                    precision_values,
                    recall_values,
                    f1_values,
                    self.statistics["support"],
                ),
            ):
                for i in range(self.num_classes):
                    self.metrics[
                        f"{metric_name}/class_{i+1:02d}"
                    ] = metric_value[i]
                if metric_name != "support":
                    self.metrics[f"{metric_name}/macro"] = metric_value.mean()
                    self.metrics[f"{metric_name}/weighted"] = (
                        metric_value * weights
                    ).sum()

            # count micro average
            self.metrics["precision/micro"] = (
                self.statistics["tp"].sum()
                / (self.statistics["tp"].sum() + self.statistics["fp"].sum())
            ).item()
            self.metrics["recall/micro"] = (
                self.statistics["tp"].sum()
                / (self.statistics["tp"].sum() + self.statistics["fn"].sum())
            ).item()
            self.metrics["f1/micro"] = (
                2
                * self.statistics["tp"].sum()
                / (
                    2 * self.statistics["tp"].sum()
                    + self.statistics["fp"].sum()
                    + self.statistics["fn"].sum()
                )
            ).item()
        return {k: self.metrics[k] for k in sorted(self.metrics.keys())}

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, f1 score and support.
        If not binary, compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics
        """
        return self.compute()
