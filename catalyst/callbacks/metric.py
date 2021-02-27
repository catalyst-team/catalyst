# @TODO: add metric aggregation, etc callback
from abc import abstractmethod, ABC
from typing import Dict, Iterable, Union, Tuple

import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics import ICallbackBatchMetric, ICallbackLoaderMetric, IMetric


class IMetricCallback(Callback, ABC):
    """Metric callback interface, abstraction over metric step."""
    @abstractmethod
    def on_loader_start(self, runner: "IRunner") -> None:
        pass

    @abstractmethod
    def on_batch_end(self, runner: "IRunner") -> None:
        pass

    @abstractmethod
    def on_loader_end(self, runner: "IRunner") -> None:
        pass


class MetricCallback(IMetricCallback):
    def __init__(
            self,
            metric: Union[ICallbackBatchMetric, ICallbackLoaderMetric],
            input_key: Union[str, Iterable[str], Dict[str, str]],
            target_key: str,
    ):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.metric = metric
        self._validate_metric(metric=metric)
        self.input_key = input_key
        self.target_key = target_key
        self._is_value_input = isinstance(input_key, str)
        self._keys = {
            **self._format_keys_to_kv(input_key),
            **self._format_keys_to_kv(target_key),
        }

    @staticmethod
    def _validate_metric(metric: IMetric) -> None:
        assert isinstance(metric, IMetric)

    @staticmethod
    def _format_keys_to_kv(keys: Union[str, Iterable[str], Dict[str, str]]) -> Dict[str, str]:
        kv_keys = {}
        if isinstance(keys, dict):
            kv_keys.update(keys)
        elif isinstance(keys, str):
            kv_keys[keys] = keys
        else:
            for key in keys:
                kv_keys[key] = key
        return kv_keys

    def _get_value_inputs(self, runner: "Runner") -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = runner.batch[self.input_key], runner.batch[self.target_key]
        inputs, targets = runner.engine.sync_tensor(inputs), runner.engine.sync_tensor(targets)
        return inputs, targets

    def _get_key_value_inputs(self, runner: "Runner") -> Dict[str, torch.Tensor]:
        kv_inputs = {}
        for key in self._keys:
            kv_inputs[self._keys[key]] = runner.engine.sync_tensor(runner.batch[key])
        return kv_inputs

    def on_batch_end(self, runner: "IRunner") -> None:
        raise NotImplementedError

    def on_loader_start(self, runner: "IRunner") -> None:
        raise NotImplementedError

    def on_loader_end(self, runner: "IRunner") -> None:
        metrics = self.metric.compute_key_value()
        runner.loader_metrics.update(metrics)


class BatchMetricCallback(MetricCallback):
    def __init__(
        self,
        metric: Union[ICallbackBatchMetric, ICallbackLoaderMetric],
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: str,
        log_on_batch: bool = True,
    ):
        super().__init__(metric=metric, input_key=input_key, target_key=target_key)
        self.log_on_batch = log_on_batch

    @staticmethod
    def _validate_metric(metric: IMetric) -> None:
        assert isinstance(metric, ICallbackBatchMetric)

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        if self._is_value_input:
            inputs, targets = self._get_value_inputs(runner=runner)
            metrics = self.metric.update_key_value(inputs, targets)
        else:
            kv_inputs = self._get_key_value_inputs(runner=runner)
            metrics = self.metric.update_key_value(**kv_inputs)

        if self.log_on_batch:
            runner.batch_metrics.update(metrics)


class LoaderMetricCallback(MetricCallback):
    @staticmethod
    def _validate_metric(metric: IMetric) -> None:
        assert isinstance(metric, ICallbackLoaderMetric)

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset(
            num_batches=runner.loader_batch_len, num_samples=runner.loader_sample_len
        )

    def on_batch_end(self, runner: "IRunner") -> None:
        if self._is_value_input:
            inputs, targets = self._get_value_inputs(runner=runner)
            self.metric.update(inputs, targets)
        else:
            kv_inputs = self._get_key_value_inputs(runner=runner)
            self.metric.update(**kv_inputs)


__all__ = ["IMetricCallback", "BatchMetricCallback", "LoaderMetricCallback"]
