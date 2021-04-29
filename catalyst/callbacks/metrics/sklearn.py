from typing import Callable, Dict, Mapping

import torch

from catalyst.callbacks.metric import FunctionalBatchMetricCallback
from catalyst.core.runner import IRunner
from catalyst.metrics import FunctionalBatchMetric


class SklearnCallback(FunctionalBatchMetricCallback):
    """@TODO: Docs."""

    def __init__(
        self,
        keys: Mapping[str, str],
        metric_fn: Callable,
        metric_key: str,
        log_on_batch: bool = True,
    ):
        """Init."""
        super().__init__(
            metric=FunctionalBatchMetric(metric_fn=metric_fn, metric_key=metric_key),
            input_key=keys,
            target_key=keys,
            log_on_batch=log_on_batch,
        )

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """@TODO: Docs."""
        kv_inputs = {}
        for key in self._keys:
            kv_inputs[key] = runner.batch[self._keys[key]].cpu().detach().numpy()
        kv_inputs["batch_size"] = runner.batch_size
        return kv_inputs
