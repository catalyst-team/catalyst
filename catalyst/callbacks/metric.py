from typing import Dict, Iterable, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics._functional_metric import FunctionalBatchMetric
from catalyst.metrics._metric import ICallbackBatchMetric, ICallbackLoaderMetric, IMetric


class IMetricCallback(Callback, ABC):
    """Metric callback interface, abstraction over metric step."""

    @abstractmethod
    def on_loader_start(self, runner: "IRunner") -> None:
        """
        On loader start action

        Args:
            runner: current runner
        """
        pass

    @abstractmethod
    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action

        Args:
            runner: current runner
        """
        pass

    @abstractmethod
    def on_loader_end(self, runner: "IRunner") -> None:
        """
        On loader end action

        Args:
            runner: current runner
        """
        pass


class MetricCallback(IMetricCallback):
    """
    MetricCallback is a base implementation of callback that updates metrics over batch or loader.

    Args:
        metric: metric to calculate in callback
        input_key: keys of tensors that should be used as inputs in metric calculation
        target_key: keys of tensors that should be used as targets in metric calculation
    """

    def __init__(
        self,
        metric: Union[ICallbackBatchMetric, ICallbackLoaderMetric],
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
    ):
        """Init MetricCallback"""
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.metric = metric
        assert isinstance(metric, IMetric)
        self._metric_update_method = self.metric.update

        kv_types = (dict, list, tuple)

        is_value_input = isinstance(input_key, str)
        is_value_targets = isinstance(target_key, str)
        is_key_value_input = isinstance(input_key, kv_types)
        is_key_value_targets = isinstance(target_key, kv_types)

        if is_value_input and is_value_targets:
            self._get_inputs = self._get_value_inputs
            self._update_metric = self._update_value_metric
        elif is_key_value_input and is_key_value_targets:
            self._get_inputs = self._get_key_value_inputs
            self._update_metric = self._update_key_value_metric
        else:
            raise NotImplementedError()

        self.input_key = input_key
        self.target_key = target_key
        self._keys = {
            **self._convert_keys_to_kv(input_key),
            **self._convert_keys_to_kv(target_key),
        }

    @staticmethod
    def _convert_keys_to_kv(keys: Union[str, Iterable[str], Dict[str, str]]) -> Dict[str, str]:
        """
        Convert keys to key-value format

        Args:
            keys: keys to convert

        Returns:
            dict of keys like {"a": "b"} where "a" is a field name of field in batch,
                "b" is a name of the same data for metric
        """
        kv_keys = {}
        if isinstance(keys, dict):
            kv_keys.update(keys)
        elif isinstance(keys, str):
            kv_keys[keys] = keys
        else:
            for key in keys:
                kv_keys[key] = key
        return kv_keys

    def _get_value_inputs(self, runner: "IRunner") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data from batch in value input case

        Args:
            runner: current runner

        Returns:
            tuple of tensor of inputs and tensor of targets
        """
        return runner.batch[self.input_key], runner.batch[self.target_key]

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """
        Get data from batch in key-value input case

        Args:
            runner: current runner

        Returns:
            dict of inputs and targets tensors
        """
        kv_inputs = {}
        for key in self._keys:
            kv_inputs[self._keys[key]] = runner.batch[key]
        return kv_inputs

    def _update_value_metric(
        self, value_inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        Update metric in value input case

        Args:
            value_inputs: tuple of input tensor and target tensor

        Returns:
            result of metric update: None or metric values
        """
        return self._metric_update_method(*value_inputs)

    def _update_key_value_metric(
        self, kv_inputs: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        Update metric in key-value input case

        Args:
            kv_inputs: input tensors in key-value format

        Returns:
            result of metric update: None or metric values
        """
        return self._metric_update_method(**kv_inputs)


class BatchMetricCallback(MetricCallback):
    """BatchMetricCallback implements batch-based metrics update and computation over loader

    Args:
        metric: metric to calculate in callback
        input_key: keys of tensors that should be used as inputs in metric calculation
        target_key: keys of tensors that should be used as targets in metric calculation
        log_on_batch: boolean flag to log computed metrics every batch
    """

    def __init__(
        self,
        metric: ICallbackBatchMetric,
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
        log_on_batch: bool = True,
    ) -> None:
        """Init BatchMetricCallback"""
        super().__init__(metric=metric, input_key=input_key, target_key=target_key)
        assert isinstance(metric, ICallbackBatchMetric)
        self.log_on_batch = log_on_batch
        self._metric_update_method = self.metric.update_key_value

    def on_loader_start(self, runner: "IRunner") -> None:
        """On loader start action: reset metric values

        Args:
            runner: current runner
        """
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action: update metric with new batch data and log it's value if necessary

        Args:
            runner: current runner
        """
        metrics_inputs = self._get_inputs(runner=runner)
        metrics = self._update_metric(metrics_inputs)
        if self.log_on_batch:
            runner.batch_metrics.update(metrics)

    def on_loader_end(self, runner: "IRunner") -> None:
        """On loader end action: compute metric values and update runner's loader metrics with it

        Args:
            runner: current runner
        """
        metrics = self.metric.compute_key_value()
        metrics = {
            k: runner.engine.sync_tensor(torch.tensor(v, device=runner.device), "mean")
            for k, v in metrics.items()
        }
        runner.loader_metrics.update(metrics)


class FunctionalBatchMetricCallback(BatchMetricCallback):
    """FunctionalBatchMetricCallback implements batch-based metrics update
    and computation over loader for ``FunctionalBatchMetric`` metrics.

    Args:
        metric: metric to calculate in callback
        input_key: keys of tensors that should be used as inputs in metric calculation
        target_key: keys of tensors that should be used as targets in metric calculation
        log_on_batch: boolean flag to log computed metrics every batch

    .. note::

        The main difference from BatchMetricCallback:
        FunctionalBatchMetricCallback also propagates current ``batch_size``
        to the FunctionalBatchMetric for correct metric computation.
    """

    def __init__(
        self,
        metric: FunctionalBatchMetric,
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
        log_on_batch: bool = True,
    ) -> None:
        """Init."""
        assert isinstance(metric, FunctionalBatchMetric)
        super().__init__(
            metric=metric, input_key=input_key, target_key=target_key, log_on_batch=log_on_batch
        )

    def _get_value_inputs(self, runner: "IRunner") -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Get data from batch in value input case

        Args:
            runner: current runner

        Returns:
            tuple of tensor of inputs and tensor of targets
        """
        return runner.batch_size, runner.batch[self.input_key], runner.batch[self.target_key]

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """Get data from batch in key-value input case

        Args:
            runner: current runner

        Returns:
            dict of inputs and targets tensors
        """
        kv_inputs = {}
        for key in self._keys:
            kv_inputs[self._keys[key]] = runner.batch[key]
        kv_inputs["batch_size"] = runner.batch_size
        return kv_inputs


class LoaderMetricCallback(MetricCallback):
    """LoaderMetricCallback implements loader-based metrics update and computation over loader

    Args:
        metric: metric to calculate in callback
        input_key: keys of tensors that should be used as inputs in metric calculation
        target_key: keys of tensors that should be used as targets in metric calculation
    """

    def __init__(
        self,
        metric: ICallbackLoaderMetric,
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
    ):
        super().__init__(metric=metric, input_key=input_key, target_key=target_key)
        assert isinstance(metric, ICallbackLoaderMetric)

    def on_loader_start(self, runner: "IRunner") -> None:
        """On loader star action: reset metric values in case of ICallbackLoaderMetric metric

        Args:
            runner: current runner
        """
        self.metric.reset(
            num_batches=runner.loader_batch_len, num_samples=runner.loader_sample_len,
        )

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action: get data from runner's batch and update metrics with it

        Args:
            runner: current runner
        """
        metrics_inputs = self._get_inputs(runner=runner)
        self._update_metric(metrics_inputs)

    def on_loader_end(self, runner: "IRunner") -> None:
        """On loader end action: compute metric values and update runner's loader metrics with it

        Args:
            runner: current runner
        """
        metrics = self.metric.compute_key_value()
        runner.loader_metrics.update(metrics)


__all__ = [
    "IMetricCallback",
    "BatchMetricCallback",
    "FunctionalBatchMetricCallback",
    "LoaderMetricCallback",
]
