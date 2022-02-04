from typing import Callable, List, Union
from functools import partial

import torch

from catalyst.core import CallbackOrder, IRunner
from catalyst.core.callback import Callback
from catalyst.metrics._accumulative import AccumulativeMetric
from catalyst.registry import REGISTRY


class SklearnModelCallback(Callback):
    """Callback to train a classifier on the train loader and
    to give predictions on the valid loader.

    Args:
        feature_key: keys of tensors that should be used as features
            for the classifier fit
        target_key: keys of tensors that should be used as targets
            for the classifier fit
        train_loader: train loader name
        valid_loaders: valid loaders where model should be predicted
        model_fn: fabric to produce objects with .fit and predict method
        predict_method: predict method name for the classifier
        predict_key: key to store computed classifier predicts in ``runner.batch``
        model_kwargs: additional parameters for ``model_fn``

    .. note::
        catalyst[ml] required for this callback
    """

    def __init__(
        self,
        feature_key: str,
        target_key: Union[str, None],
        train_loader: str,
        valid_loaders: Union[str, List[str]],
        model_fn: Union[Callable, str],
        predict_method: str = "predict",
        predict_key: str = "sklearn_predict",
        **model_kwargs,
    ) -> None:
        super().__init__(order=CallbackOrder.Metric)

        if isinstance(model_fn, str):
            model_fn = REGISTRY.get(model_fn)

        assert hasattr(
            model_fn(), predict_method
        ), "The classifier must have the predict method!"

        self._train_loader = train_loader
        if isinstance(valid_loaders, str):
            self._valid_loaders = [valid_loaders]
        else:
            self._valid_loaders = valid_loaders
        self.model_fabric_fn = partial(model_fn, **model_kwargs)
        self.feature_key = feature_key
        self.target_key = target_key
        self.predict_method = predict_method
        self.predict_key = predict_key
        self.model = None

        if self.target_key:
            self.storage = AccumulativeMetric(keys=[feature_key, target_key])
        if self.target_key is None:
            self.storage = AccumulativeMetric(keys=[feature_key])

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Loader start hook: initiliaze storages for the loaders.

        Args:
            runner: current runner
        """
        super().on_loader_start(runner)
        if runner.loader_key == self._train_loader:
            self.storage.reset(
                num_samples=runner.loader_sample_len, num_batches=runner.loader_batch_len
            )
        if runner.loader_key in self._valid_loaders:
            assert self.model is not None, "The train loader has to be processed first!"

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action: get data from runner's batch
        and update a loader storage with it

        Args:
            runner: runner for the experiment.
        """
        assert (
            torch.isnan(runner.batch[self.feature_key]).sum() == 0
        ), "SklearnModelCallback can't process Tensors with NaN!"
        if runner.loader_key == self._train_loader:
            self.storage.update(**runner.batch)
        if runner.loader_key in self._valid_loaders:
            features = runner.batch[self.feature_key].detach().cpu().numpy()
            # classifier predict
            classifier_predict = getattr(self.model, self.predict_method)
            predictions = classifier_predict(features)
            runner.batch[self.predict_key] = torch.tensor(
                predictions, device=runner.engine.device
            )

    def on_loader_end(self, runner: "IRunner") -> None:
        """Loader end hook: for the train loader train classifier,
        for the test check the quality.

        Args:
            runner: current runner
        """
        if runner.loader_key == self._train_loader:
            data = self.storage.compute_key_value()
            collected_size = self.storage.collected_samples
            loader_len = runner.loader_sample_len

            assert (
                collected_size == loader_len
            ), f"collected samples - {collected_size} != loader len - {loader_len}!"

            assert (
                torch.isnan(data[self.feature_key]).sum() == 0
            ), "SklearnModelCallback - NaN after Accumulation!"

            self.model = self.model_fabric_fn()
            if self.target_key is None:
                features = data[self.feature_key].detach().cpu().numpy()
                self.model.fit(features)
            else:
                features = data[self.feature_key].detach().cpu().numpy()
                targets = data[self.target_key].detach().cpu().numpy()
                self.model.fit(features, targets)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        Epoch end hook: the callback delete the model.

        Args:
            runner: current runner
        """
        # We need this for the control of a loader order.
        self.model = None


__all__ = ["SklearnModelCallback"]
