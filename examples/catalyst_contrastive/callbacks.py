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


# class FeatureAccumulator(AccumulationMetric):
#     """Feature bank,
#     Args:
#         prefix: embeddings accumulator prefix
#         suffix: embeddings accumulator suffix
#     """

#     def __init__(self, save_path, prefix: str = None, suffix: str = None):
#         """Init."""
#         super().__init__(compute_on_call=False, prefix=prefix, suffix=suffix)
#         self.accaulator_name = f"{self.prefix}feature_accumulator{self.suffix}"
#         self.features = []
#         self.targets = []
#         self.save_path = save_path

#     def reset(self, loader_name, epoch) -> None:
#         """Resets all fields"""
#         self.features = []
#         self.targets = []
#         self.loader_name = loader_name
#         self.epoch = epoch

#     def update(self, features: torch.tensor, targets: torch.Tensor) -> None:
#         """Updates accumulator with features and targets for new data.
#         Args:
#             features: tensor with features
#             targets: tensor with targets
#         """
#         self.features.append(features.cpu().detach())
#         self.targets.append(targets.cpu().detach())

#     def compute(self) -> Tuple[torch.Tensor, float, float, float]:
#         """Concat computed features and targets."""
#         targets = torch.cat(self.targets)
#         features = torch.cat(self.features)
#         return features, targets

#     def compute_key_value(self) -> Dict[str, float]:
#         """Save computed features with targets."""
#         features, targets = self.compute()

#         feature_bank = {"feature": features, "targets": targets}

#         file_name = f"{self.loader_name}_{self.epoch}_embs.pkl"
#         file_name = Path(self.save_path) / file_name
#         with open(file_name, "wb") as handle:
#             pickle.dump(feature_bank, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return {}


# class FeatureAccumulatorCallback(LoaderMetricCallback):
#     """Feature accumulator callback.
#     Args:
#         input_key: input key to use for features, specifies our ``features``.
#         target_key: output key to use for targets, specifies our ``y_true``.
#         prefix: accumulator prefix
#         suffix: accumulator suffix
#     """

#     def __init__(
#         self, save_path, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
#     ):
#         """Init."""
#         super().__init__(
#             metric=FeatureAccumulator(save_path, prefix=prefix, suffix=suffix),
#             input_key=input_key,
#             target_key=target_key,
#         )

#     def on_loader_start(self, runner) -> None:
#         """On loader star action: reset metric values in case of ICallbackLoaderMetric metric

#         Args:
#             runner: current runner
#         """
#         self.metric.reset(loader_name=runner.loader_key, epoch=runner.global_epoch_step)


# class KNNClassiferCallback(Callback):
#     """
#     Classify your batch with weighted KNN classifier.
#     Args:
#         input_key str: KNN classifier features key in a batch
#         output_key str: Keys for output (probs).
#     Raises:
#         TypeError: When keys is not str or a list.
#     """

#     def __init__(
#         self,
#         feature_bank_path: str,
#         input_key: str,
#         output_key: Union[List[str], str] = None,
#         k: int = 200,
#         temperature: float = 0.5,
#         device: str = "cuda",
#     ):
#         """
#         Preprocess your batch with specified function.
#         Args:
#             input_key str: Keys in batch dict to features for classification.
#             output_key str: Keys for output (logits).
#             k int: Top k most similar images used to predict the label.
#             temperatire float: Temperature used in softmax.
#         Raises:
#             TypeError: When keys is not str or a list.
#         """
#         super().__init__(order=CallbackOrder.Internal)

#         if not isinstance(feature_bank_path, str):
#             raise TypeError("feature bank path should be str.")

#         if not isinstance(input_key, str):
#             raise TypeError("input key should be str.")
#         self._handle_batch = self._handle_value

#         if not isinstance(output_key, str):
#             raise TypeError("output key should be str.")

#         self.input_key = input_key
#         self.output_key = output_key
#         self.feature_bank_path = feature_bank_path
#         self.k = k
#         self.temperature = temperature
#         self.device = device

#     def on_loader_start(self, runner: "IRunner") -> None:
#         with open(self.feature_bank_path, "rb") as handle:
#             self.feature_bank = pickle.load(handle)
#         self.feature_bank["feature"] = self.feature_bank["feature"].to(
#             self.device, non_blocking=True
#         )
#         # self.feature_bank['targets'] = self.feature_bank['targets'].to(self.device, non_blocking=True)

#     def _handle_value(self, runner):
#         batch_in = runner.batch[self.input_key]
#         batch_in = batch_in.to(self.device)

#         feature, feature_bank = batch_in, self.feature_bank["feature"]
#         feature_labels = self.feature_bank["targets"]

#         number_of_classes = len(Counter(self.feature_bank["targets"]))

#         sim_matrix = torch.mm(feature, feature_bank.t()).detach().cpu()
#         # [B, K]
#         # not enough cuda memory maybe can be fixed (calculations on cpu now)
#         sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
#         # [B, K]
#         sim_labels = torch.gather(
#             feature_labels.expand(batch_in.size(0), -1), dim=-1, index=sim_indices
#         )
#         sim_weight = (sim_weight / self.temperature).exp()

#         # counts for each class
#         one_hot_label = torch.zeros(
#             batch_in.size(0) * self.k, number_of_classes, device=sim_labels.device
#         )
#         # [B*K, C]

#         # the memory isshue is here
#         one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#         # weighted score ---> [B, C]
#         pred_scores = torch.sum(
#             one_hot_label.view(batch_in.size(0), -1, number_of_classes)
#             * sim_weight.unsqueeze(dim=-1),
#             dim=1,
#         )
#         pred_scores = pred_scores / pred_scores.sum(dim=1).unsqueeze(-1)

#         runner.batch.update(**{self.output_key: pred_scores})

#     def on_batch_end(self, runner: "IRunner") -> None:
#         """On batch end action.
#         Args:
#             runner: runner for the experiment.
#         """
#         self._handle_batch(runner)
