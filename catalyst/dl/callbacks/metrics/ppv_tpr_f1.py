from typing import List
from collections import defaultdict

import numpy as np
import torch

from catalyst.dl.meters import PrecisionRecallF1ScoreMeter
from catalyst.dl.core import MeterMetricsCallback, RunnerState, CallbackOrder
from catalyst.utils import get_activation_fn


class PrecisionRecallF1ScoreCallback(MeterMetricsCallback):
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
        threshold: float = 0.5,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for metric calculation
                specifies our ``y_true``.
            output_key (str): output key to use for metric calculation;
                specifies our ``y_pred``
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)
        assert self.num_classes is not None

        meters = [PrecisionRecallF1ScoreMeter(threshold)
                       for _ in range(self.num_classes)]

        super().__init__(
            metric_names=["ppv", "tpr", "f1"],
            meter_list=meters,
            input_key=input_key,
            output_key=output_key,
            class_names=class_names,
            activation=activation
        )


__all__ = ["PrecisionRecallF1ScoreCallback"]
