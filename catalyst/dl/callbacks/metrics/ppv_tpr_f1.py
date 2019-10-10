from typing import List
from collections import defaultdict

import numpy as np
import torch

from catalyst.dl.meters import PrecisionRecallF1ScoreMeter
from catalyst.dl.core import Callback, RunnerState, CallbackOrder


class PrecisionRecallF1ScoreCallback(Callback):
    """
    Calculates the global precision (positive predictive value or ppv),
    recall (true positive rate or tpr), and F1-score per class for each loader.
    Currently, supports binary and multi-label cases.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        class_names: List[str] = None,
        num_classes: int = 1,
        threshold: float = 0.5
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes
            threshold (float): threshold for outputs binarization
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key

        self.list_args = ["ppv", "tpr", "f1-score"]
        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None

        self.meters = [PrecisionRecallF1ScoreMeter(threshold)
                       for _ in range(self.num_classes)]

    def _reset_stats(self):
        for meter in self.meters:
            meter.reset()

    def on_loader_start(self, state):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)

        if self.num_classes == 1 and len(probabilities.shape) == 1:
            self.meters[0].add(probabilities, targets)
        else:
            for i in range(self.num_classes):
                self.meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: RunnerState):
        prec_recall_f1score = defaultdict(list)
        loader_values = state.metrics.epoch_values[state.loader_name]
        for i, meter in enumerate(self.meters):
            metrics = meter.value()
            postfix = self.class_names[i] \
                if self.class_names is not None \
                else str(i)
            for prefix, metric_ in zip(self.list_args, metrics):
                # adding the per-class metrics
                prec_recall_f1score[prefix] = metric_
                metric_name = f"{prefix}/class_{postfix}"
                loader_values[metric_name] = metric_

        for prefix in self.list_args:
            # averages computed metrics
            mean_value = float(np.mean(prec_recall_f1score[prefix]))
            metric_name = f"{prefix}/_mean"
            loader_values[metric_name] = mean_value

        self._reset_stats()


__all__ = ["PrecisionRecallF1ScoreCallback"]
