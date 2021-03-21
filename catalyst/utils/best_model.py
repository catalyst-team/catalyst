from typing import Any, Dict, Mapping

from catalyst.typing import Model


class BestModel:
    def __init__(self, metric: str, minimize_metric: bool = False):
        """
        Class for store best model state dict.

        Args:
            metric: metric to choose best model.
            minimize_metric: minimize/maximize metric.
        """
        self.metric = metric
        self.minimize_metric = minimize_metric
        self._best_model_sd = None
        self._best_metric = None

    def add_result(self, epoch_metrics: Mapping[str, Any], model: Model) -> None:
        """
        Adds result for current epoch and saves state dict if current epoch is best.
        Args:
            epoch_metrics: dict of metrics for epoch
            model: current model

        Raises:
            Exception: if specified metric not in epoch metrics dict.
        """
        if self.metric not in epoch_metrics.keys():
            raise Exception(f"Metric {self.metric} not in runner.epoch_metrics.")
        current_metric = epoch_metrics[self.metric]
        if self._best_metric is None:
            self._best_metric = current_metric
            self._best_model_sd = model.state_dict()
        else:
            is_best_model = (self.minimize_metric and self._best_metric > current_metric) or (
                not self.minimize_metric and self._best_metric < current_metric
            )
            if is_best_model:
                self._best_metric = current_metric
                self._best_model_sd = model.state_dict()

    def get_best_model_sd(self) -> Dict[str, Any]:
        """
        Gets best model state dict.

        Returns: state dict.
        """
        if self._best_model_sd is None:
            raise Exception(f"There is no best model.")
        return self._best_model_sd


__all__ = ["BestModel"]
