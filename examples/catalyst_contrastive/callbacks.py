from typing import Any, Callable, Dict, List, Tuple, Union
from collections import Counter
from functools import partial
from pathlib import Path
import pickle

import torch

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.contrib.nn.modules import se
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.core.callback import ICallback
from catalyst.metrics._metric import AccumulationMetric, ICallbackLoaderMetric
from catalyst.registry import REGISTRY


class SklearnClassifierCallback(ICallback):
    def __init__(
        self,
        feautres_key: str,
        targets_key: str,
        train_loader: str,
        valid_loader: str,
        sklearn_classifier_fn: Callable,
        sklearn_metric_fn: Callable,
    ) -> None:
        super().__init__()
        self.order = CallbackOrder.Internal
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self.classifier_fabric = sklearn_classifier_fn
        self.metric_fn = sklearn_metric_fn
        self.feature_key = feautres_key
        self.target_key = targets_key
        self.storage = {
            self._train_loader: AccumulationMetric(
                accumulative_fields=[feautres_key, targets_key]
            ),
            self._valid_loader: AccumulationMetric(
                accumulative_fields=[feautres_key, targets_key]
            ),
        }
        self.classifier = None

    def on_loader_start(self, runner: "IRunner") -> None:
        super().on_loader_start(runner)
        if runner.loader_key == self._train_loader:
            self.storage[self._train_loader].reset(
                num_samples=runner.loader_batch_size * runner.loader_batch_len,
                num_batches=runner.loader_batch_len,
            )
        if runner.loader_key == self._valid_loader:
            self.storage[self._valid_loader].reset(
                num_samples=runner.loader_batch_size * runner.loader_batch_len,
                num_batches=runner.loader_batch_len,
            )

    def on_loader_end(self, runner: "IRunner") -> None:
        if runner.loader_key == self._train_loader:
            data = self.storage[self._train_loader].compute_key_value()
            # classifier fit
            X, y = data[self.feature_key].numpy(), data[self.target_key].numpy()
            self.classfier = self.classifier_fabric()
            self.classfier.fit(X, y)
        if runner.loader_key == self._valid_loader:
            data = self.storage[self._train_loader].compute_key_value()
            X, y = data[self.feature_key], data[self.target_key]
            y_pred = self.classfier.predict_proba(data[self.feature_key])
            metric_val = self.metric_fn(y, y_pred, k=1)
            runner.loader_metrics.update({"sklear_classifier_metric": metric_val})
            return metric_val

    def on_batch_end(self, runner: "IRunner") -> None:
        if runner.loader_key in self.storage:
            loader_storage = self.storage[runner.loader_key]
            loader_storage.update(**runner.batch)