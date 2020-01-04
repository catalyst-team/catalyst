from typing import Callable, List  # isort:skip
from collections import defaultdict

import numpy as np

from catalyst.core import CallbackOrder, Callback, LoggerCallback
from catalyst.utils import get_activation_fn
from .state import DLRunnerState


class MeterMetricsCallback(Callback):
    """
    A callback that tracks metrics through meters and prints metrics for
    each class on `state.on_loader_end`.
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
                len(meter_list) == n_classes
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
        self._reset_stats()

    def on_batch_end(self, state: State):
        logits = state.output[self.output_key].detach().float()
        targets = state.input[self.input_key].detach().float()
        probabilities = self.activation_fn(logits)

        for i in range(self.num_classes):
            self.meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: State):
        metrics_tracker = defaultdict(list)
        loader_values = state.metrics.epoch_values[state.loader_name]
        # Computing metrics for each class
        for i, meter in enumerate(self.meters):
            metrics = meter.value()
            postfix = self.class_names[i] \
                if self.class_names is not None \
                else str(i)
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


class MetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: State):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]
        metric = self.metric_fn(outputs, targets, **self.metric_params)
        state.metrics.add_batch_value(name=self.prefix, value=metric)


class MultiMetricCallback(Callback):
    """
    A callback that returns multiple metrics on `state.on_batch_end`
    """

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        list_args: List,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.list_args = list_args
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: State):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        metrics_ = self.metric_fn(
            outputs, targets, self.list_args, **self.metric_params
        )

        batch_metrics = {}
        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            batch_metrics[key] = metric
        state.metrics.add_batch_value(metrics_dict=batch_metrics)


__all__ = [
    "CallbackOrder",
    "Callback",
    "LoggerCallback",
    "MeterMetricsCallback",
    "MetricCallback",
    "MultiMetricCallback",
]
