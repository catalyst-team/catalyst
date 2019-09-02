from typing import List

import numpy as np
import torch

from catalyst.dl.meters import AUCMeter
from catalyst.dl.core import Callback, RunnerState, CallbackOrder


class AUCCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc",
        class_names: List[str] = None,
        num_classes: int = 1
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None

        self.auc_meters = [AUCMeter() for _ in range(self.num_classes)]

    def _reset_stats(self):
        for auc_meter in self.auc_meters:
            auc_meter.reset()

    def on_loader_start(self, state):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)

        if self.num_classes == 1 and len(probabilities.shape) == 1:
            self.auc_meters[0].add(probabilities, targets)
        else:
            for i in range(self.num_classes):
                self.auc_meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: RunnerState):
        areas = []

        for i, auc_meter in enumerate(self.auc_meters):
            area, _, _ = auc_meter.value()
            area = float(area)
            postfix = self.class_names[i] \
                if self.class_names is not None \
                else str(i)
            metric_name = f"{self.prefix}/class_{postfix}"
            state.metrics.epoch_values[state.loader_name][metric_name] = area
            areas.append(area)

        area = float(np.mean(areas))
        metric_name = f"{self.prefix}/_mean"
        state.metrics.epoch_values[state.loader_name][metric_name] = area

        self._reset_stats()


__all__ = ["AUCCallback"]
