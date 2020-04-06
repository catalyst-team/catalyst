from typing import List
from collections import defaultdict

import numpy as np

from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl.utils import get_activation_fn


class MeterMetricsCallback(Callback):
    """
    A callback that tracks metrics through meters and prints metrics for
    each class on `state.on_loader_end`.

    .. note::
        This callback works for both single metric and multi-metric meters.
    """

    def __init__(
        self,
        metric_names: List[str],
        meter_list: List,
        input_key: str = "targets",
        output_key: str = "logits",
        class_names: List[str] = None,
        num_classes: int = 2,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            metric_names (List[str]): of metrics to print
                Make sure that they are in the same order that metrics
                are outputted by the meters in `meter_list`
            meter_list (list-like): List of meters.meter.Meter instances
                len(meter_list) == num_classes
            input_key (str): input key to use for metric calculation
                specifies our ``y_true``.
            output_key (str): output key to use for metric calculation;
                specifies our ``y_pred``
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes; must be > 1
            activation (str): An torch.nn activation applied to the logits.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(CallbackOrder.Metric)
        self.metric_names = metric_names
        self.meters = meter_list
        self.input_key = input_key
        self.output_key = output_key
        self.class_names = class_names
        self.num_classes = num_classes
        self.activation = activation
        self.activation_fn = get_activation_fn(self.activation)

    def _reset_stats(self):
        for meter in self.meters:
            meter.reset()

    def on_loader_start(self, state):
        """Loader start hook.

        Args:
            state (State): current state
        """
        self._reset_stats()

    def on_batch_end(self, state: State):
        """Batch end hook. Computes batch metrics.

        Args:
            state (State): current state
        """
        logits = state.batch_out[self.output_key].detach().float()
        targets = state.batch_in[self.input_key].detach().float()
        probabilities = self.activation_fn(logits)

        for i in range(self.num_classes):
            self.meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: State):
        """Loader end hook. Computes loader metrics.

        Args:
            state (State): current state
        """
        metrics_tracker = defaultdict(list)
        loader_values = state.loader_metrics
        # Computing metrics for each class
        for i, meter in enumerate(self.meters):
            metrics = meter.value()
            postfix = (
                self.class_names[i] if self.class_names is not None else str(i)
            )
            for prefix, metric_ in zip(self.metric_names, metrics):
                # appending the per-class values
                metrics_tracker[prefix].append(metric_)
                metric_name = f"{prefix}/class_{postfix}"
                loader_values[metric_name] = metric_
        # averaging the per-class values for each metric
        for prefix in self.metric_names:
            mean_value = float(np.mean(metrics_tracker[prefix]))
            metric_name = f"{prefix}/_mean"
            loader_values[metric_name] = mean_value

        self._reset_stats()


__all__ = ["MeterMetricsCallback"]
