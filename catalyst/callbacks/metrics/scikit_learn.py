from typing import Any, Callable, Dict, Mapping, Union

import torch

from catalyst.callbacks.metric import FunctionalBatchMetricCallback
from catalyst.core.runner import IRunner
from catalyst.metrics import FunctionalBatchMetric
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    import sklearn


class SklearnCallback(FunctionalBatchMetricCallback):
    """

    Args:
        keys:
        metric_fn:
        metric_key:
        log_on_batch:
    """

    def __init__(
        self,
        keys: Mapping[str, Any],
        metric_fn: Union[Callable, str],
        metric_key: str,
        log_on_batch: bool = True,
    ):
        """Init."""
        if isinstance(metric_fn, str):
            metric_fn = sklearn.metrics.__dict__[metric_fn]

        super().__init__(
            metric=FunctionalBatchMetric(metric_fn=metric_fn, metric_key=metric_key),
            input_key=keys,
            target_key=keys,
            log_on_batch=log_on_batch,
        )

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        kv_inputs = {}
        for key, value in self._keys.items():
            if value in runner.batch:
                kv_inputs[key] = runner.batch[value].cpu().detach().numpy()
            else:
                kv_inputs[key] = self._keys[key]
        kv_inputs["batch_size"] = runner.batch_size
        return kv_inputs
