from typing import List

import numpy as np
import torch

from catalyst.dl.meters import AUCMeter
from catalyst.dl.core import MeterMetricsCallback, RunnerState


class AUCCallback(MeterMetricsCallback):
    """
    Calculates the AUC  per class for each loader.
    Currently, supports binary and multi-label cases.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc",
        class_names: List[str] = None,
        num_classes: int = 1,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for auc calculation
                specifies our ``y_true``.
            output_key (str): output key to use for auc calculation;
                specifies our ``y_pred``
            prefix (str): name to display for auc when printing
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)
        assert self.num_classes is not None

        meters = [AUCMeter() for _ in range(self.num_classes)]

        super().__init__(
            metric_names=[prefix],
            meter_list=meters,
            input_key=input_key,
            output_key=output_key,
            class_names=class_names,
            activation=activation
        )


__all__ = ["AUCCallback"]
