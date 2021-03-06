# @TODO: add metric aggregation, etc callback
from typing import Dict, Iterable, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics import (
    ICallbackBatchMetric,
    ICallbackLoaderMetric,
    IMetric,
)


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
    def __init__(
        self,
        metric: Union[ICallbackBatchMetric, ICallbackLoaderMetric],
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
    ):
        """
        Init MetricCallback

        Args:
            metric: metric to calculate in callback
            input_key: keys of tensors that should be used as inputs in metric calculation
            target_key: keys of tensors that should be used as targets in metric calculation
        """
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.metric = metric
        self._validate_metric(metric=metric)
        self._metric_update_method = self.metric.update

        kv_types = (dict, tuple, list)

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
    def _validate_metric(metric: IMetric) -> None:
        """
        Check if given metric can be used in this callback

        Args:
            metric: metric to check
        """
        assert isinstance(metric, IMetric)

    @staticmethod
    def _convert_keys_to_kv(
        keys: Union[str, Iterable[str], Dict[str, str]]
    ) -> Dict[str, str]:
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

    def _get_value_inputs(
        self, runner: "IRunner"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data from batch in value input case

        Args:
            runner: current runner

        Returns:
            tuple of tensor of inputs and tensor of targets
        """
        inputs, targets = (
            runner.batch[self.input_key],
            runner.batch[self.target_key],
        )
        inputs, targets = (
            runner.engine.sync_tensor(inputs),
            runner.engine.sync_tensor(targets),
        )
        return inputs, targets

    def _get_key_value_inputs(
        self, runner: "IRunner"
    ) -> Dict[str, torch.Tensor]:
        """
        Get data from batch in key-value input case

        Args:
            runner: current runner

        Returns:
            dict of inputs and targets tensors
        """
        kv_inputs = {}
        for key in self._keys:
            kv_inputs[self._keys[key]] = runner.engine.sync_tensor(
                runner.batch[key]
            )
        return kv_inputs

    def _update_value_metric(
        self, inputs_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        Update metric in value input case

        Args:
            inputs_tuple: tuple of input tensor and target tensor

        Returns:
            result of metric update: None or metric values
        """
        inputs, targets = inputs_tuple
        return self._metric_update_method(inputs, targets)

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

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        On loader start action: reset metric values

        Args:
            runner: current runner
        """
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action: get data from runner's batch and update metrics with it

        Args:
            runner: current runner
        """
        metrics_inputs = self._get_inputs(runner=runner)
        return self._update_metric(metrics_inputs)

    def on_loader_end(self, runner: "IRunner") -> None:
        """
        On loader end action: compute metric values and update runner's loader metrics with it

        Args:
            runner: current runner
        """
        metrics = self.metric.compute_key_value()
        runner.loader_metrics.update(metrics)


class BatchMetricCallback(MetricCallback):
    def __init__(
        self,
        metric: Union[ICallbackBatchMetric, ICallbackLoaderMetric],
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
        log_on_batch: bool = True,
    ) -> None:
        """
        Init BatchMetricCallback

        Args:
            metric: metric to calculate in callback
            input_key: keys of tensors that should be used as inputs in metric calculation
            target_key: keys of tensors that should be used as targets in metric calculation
            log_on_batch: if True update runner's batch metrics every batch
        """
        super().__init__(
            metric=metric, input_key=input_key, target_key=target_key
        )
        self.log_on_batch = log_on_batch
        self._metric_update_method = self.metric.update_key_value

    @staticmethod
    def _validate_metric(metric: IMetric) -> None:
        """
        Check if metric is an instance of ICallbackBatchMetric and can be used in
            BatchMetricCallback

        Args:
            metric: metric to check
        """
        assert isinstance(metric, ICallbackBatchMetric)

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action: update metric with new batch data and log it's value if necessary

        Args:
            runner: current runner
        """
        metrics = super().on_batch_end(runner=runner)
        if self.log_on_batch:
            runner.batch_metrics.update(metrics)


class LoaderMetricCallback(MetricCallback):
    @staticmethod
    def _validate_metric(metric: IMetric) -> None:
        """
        Check if metric is an instance of ICallbackLoaderMetric and can be used in
            LoaderMetricCallback

        Args:
            metric: metric to check
        """
        assert isinstance(metric, ICallbackLoaderMetric)

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        On loader star action: reset metric values in case of ICallbackLoaderMetric metric

        Args:
            runner: current runner
        """
        self.metric.reset(
            num_batches=runner.loader_batch_len,
            num_samples=runner.loader_sample_len,
        )


__all__ = [
    "IMetricCallback",
    "BatchMetricCallback",
    "LoaderMetricCallback",
]
