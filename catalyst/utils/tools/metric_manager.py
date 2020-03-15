from typing import Any, Dict  # isort:skip
from collections import defaultdict
from numbers import Number

from catalyst import utils
from catalyst.utils import meters


class MetricManager:
    @staticmethod
    def _to_single_value(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()

        assert isinstance(value, Number), \
            f"{type(value)} is not a python number"

        value = float(value)
        return value

    def __init__(
        self,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize: bool = True,
        batch_consistant_metrics: bool = True,
    ):
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize = minimize
        self._batch_consistant_metrics = batch_consistant_metrics

        self._meters: Dict[str, meters.AverageValueMeter] = None
        self._batch_values: Dict[str, float] = None
        self.epoch_values: Dict[str, Dict[str:float]] = None
        self.valid_values: Dict[str, float] = None

        self.best_main_metric_value: float = \
            float("+inf") if self._minimize else float("-inf")
        self.is_best: bool = None

        self._current_loader_name: str = None

    @property
    def batch_values(self) -> Dict[str, float]:
        self.add_batch_value()
        return self._batch_values

    @property
    def main_metric_value(self) -> float:
        assert self._valid_loader in self.epoch_values, \
            f"{self._valid_loader} is not available yet"
        assert self._main_metric in self.epoch_values[self._valid_loader], \
            f"{self._main_metric} is not available yet"

        return self.epoch_values[self._valid_loader][self._main_metric]

    def begin_epoch(self):
        self.epoch_values = defaultdict(lambda: {})

    def end_epoch_train(self):
        assert self._valid_loader in self.epoch_values, \
            f"{self._valid_loader} is not available by the epoch end"
        assert self._main_metric in self.epoch_values[self._valid_loader], \
            f"{self._main_metric} value is not available by the epoch end"

        self.valid_values = self.epoch_values[self._valid_loader]

        if self._minimize:
            self.is_best = self.main_metric_value < self.best_main_metric_value
        else:
            self.is_best = self.main_metric_value > self.best_main_metric_value

        if self.is_best:
            self.best_main_metric_value = self.main_metric_value

    def begin_loader(self, name: str):
        self._current_loader_name = name
        self._meters = defaultdict(meters.AverageValueMeter)

    def end_loader(self):
        for name, meter in self._meters.items():
            self.epoch_values[self._current_loader_name][name] = meter.mean

        self._current_loader_name = None

    def begin_batch(self):
        self._batch_values = {}

    def end_batch(self):
        if len(self._meters) != 0 and self._batch_consistant_metrics:
            assert self._meters.keys() == self._batch_values.keys(), \
                "Metric set is not consistent among batches"

        for name, value in self._batch_values.items():
            self._meters[name].add(value)

    def add_batch_value(
        self,
        name: str = None,
        value: Any = None,
        metrics_dict: Dict[str, Any] = None
    ):
        metrics_dict = metrics_dict or {}

        if name:
            metrics_dict[name] = value

        for metric_name, metric_value in metrics_dict.items():
            metric_value = self._to_single_value(metric_value)
            metric_value = utils.distributed_mean(metric_value)
            self._batch_values[metric_name] = metric_value


__all__ = ["MetricManager"]
