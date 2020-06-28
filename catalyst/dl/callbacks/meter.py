from typing import List
from collections import defaultdict

import numpy as np

from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl.utils import get_activation_fn


class MeterMetricsCallback(Callback):
    """
    A callback that tracks metrics through meters and prints metrics for
    each class on `runner.on_loader_end`.

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
        super().__init__(CallbackOrder.metric)
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

    def on_loader_start(self, runner: IRunner):
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        self._reset_stats()

    def on_batch_end(self, runner: IRunner):
        """Batch end hook. Computes batch metrics.

        Args:
            runner (IRunner): current runner
        """
        logits = runner.output[self.output_key].detach().float()
        targets = runner.input[self.input_key].detach().float()
        probabilities = self.activation_fn(logits)

        for i in range(self.num_classes):
            self.meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, runner: IRunner):
        """Loader end hook. Computes loader metrics.

        Args:
            runner (IRunner): current runner
        """
        metrics_tracker = defaultdict(list)
        loader_values = runner.loader_metrics
        # Computing metrics for each class
        for i, meter in enumerate(self.meters):
            metrics = meter.value()
            postfix = (
                self.class_names[i] if self.class_names is not None else str(i)
            )
            for prefix, metric in zip(self.metric_names, metrics):
                # appending the per-class values
                metrics_tracker[prefix].append(metric)
                metric_name = f"{prefix}/class_{postfix}"
                loader_values[metric_name] = metric
        # averaging the per-class values for each metric
        for prefix2 in self.metric_names:
            mean_value = float(np.mean(metrics_tracker[prefix2]))
            metric_name = f"{prefix2}/_mean"
            loader_values[metric_name] = mean_value

        self._reset_stats()


__all__ = ["MeterMetricsCallback"]
