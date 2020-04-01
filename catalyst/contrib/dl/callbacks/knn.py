from typing import Dict, List
from math import ceil

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors

import torch

from catalyst.dl import Callback, CallbackOrder, State


class KNNMetricCallback(Callback):
    """A callback that returns single metric on ``state.on_loader_end``."""

    def __init__(
        self,
        input_key: str = "logits",
        output_key: str = "targets",
        prefix: str = "knn",
        num_classes: int = 2,
        class_names: dict = None,
        cv_loader_names: Dict[str, List[str]] = None,
        metric_fn: str = "f1-score",
        knn_metric: str = "euclidean",
        num_neighbors: int = 5,
    ):
        """Returns metric value calculated using kNN algorithm.

        Args:
            input_key: input key to get features.
            output_key: output key to get targets.
            prefix: key to store in logs.
            num_classes: Number of classes; must be > 1.
            class_names: of indexes and class names.
            cv_loader_names: dict with keys and values of loader_names
                for which cross validation should be calculated.
                For example {"train" : ["valid", "test"]}.
            metric_fn: one of `accuracy`, `precision`, `recall`, `f1-score`.
                       default is `f1-score`.
            knn_metric: look sklearn.neighbors.NearestNeighbors parameter.
            num_neighbors: number of neighbors, default is 5.
        """
        super().__init__(CallbackOrder.Metric)

        assert num_classes > 1, "`num_classes` should be more than 1"

        metric_fns = {
            "accuracy": accuracy_score,
            "recall": recall_score,
            "precision": precision_score,
            "f1-score": f1_score,
        }

        assert (
            metric_fn in metric_fns
        ), f"Metric function with value `{metric_fn}` not implemented"

        self.prefix = prefix
        self.features_key = input_key
        self.targets_key = output_key

        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names is not None
            else [str(i) for i in range(num_classes)]
        )

        self.cv_loader_names = cv_loader_names

        self.metric_fn = metric_fns[metric_fn]
        self.knn_metric = knn_metric
        self.num_neighbors = num_neighbors

        self.num_folds = 1

        self._reset_cache()
        self._reset_sets()

    def _reset_cache(self):
        """Function to reset cache for features and labels."""
        self.features = []
        self.targets = []

    def _reset_sets(self):
        """Function to reset cache for all sets."""
        self.sets = {}

    def _knn(self, train_set, test_set=None):
        """Returns accuracy calculated using kNN algorithm.

        Args:
            train_set: dict of feature "values" and "labels" for training set.
            test_set: dict of feature "values" and "labels" for test set.

        Returns:
            cm: tuple of lists of true & predicted classes.
        """
        # if the test_set is None, we will test train_set on itself,
        # in that case we need to delete the closest neighbor
        leave_one_out = test_set is None

        if leave_one_out:
            test_set = train_set

        x_train, y_train = train_set["values"], train_set["labels"]
        x_test, y_test = test_set["values"], test_set["labels"]

        size = len(y_train)

        result = None
        while result is None:
            try:
                y_pred = []

                # fit nearest neighbors class on our train data
                classifier = NearestNeighbors(
                    num_neighbors=self.num_neighbors + int(leave_one_out),
                    metric=self.knn_metric,
                    algorithm="brute",
                )
                classifier.fit(x_train, y_train)

                # data could be evaluated in num_folds in order to avoid OOM
                end_idx, batch_size = 0, ceil(size / self.num_folds)
                for start_idx in range(0, size, batch_size):

                    end_idx = min(start_idx + batch_size, size)

                    x = x_test[start_idx:end_idx]

                    knn_ids = classifier.kneighbors(x, return_distance=False)

                    # if we predict train set on itself we have to delete 0th
                    # neighbor for all of the distances
                    if leave_one_out:
                        knn_ids = knn_ids[:, 1:]

                    # calculate the most frequent class across k neighbors
                    knn_classes = y_train[knn_ids]
                    knn_classes, _ = stats.mode(knn_classes, axis=1)

                    y_pred.extend(knn_classes[:, 0].tolist())

                y_pred = np.asarray(y_pred)

                result = (y_test, y_pred)

            # this try catch block made because sometimes sets are quite big
            # and it is not possible to put everything in memory, so we split
            except MemoryError:
                print(
                    f"Memory error with {self.num_folds} folds, trying more."
                )
                self.num_folds *= 2
                result = None

        return result

    def on_batch_end(self, state: State) -> None:
        """Batch end hook.

        Args:
            state (State): current state
        """
        features: torch.Tensor = state.batch_out[
            self.features_key
        ].cpu().detach().numpy()
        targets: torch.Tensor = state.batch_in[
            self.targets_key
        ].cpu().detach().numpy()

        self.features.extend(features)
        self.targets.extend(targets)

    def on_loader_end(self, state: State) -> None:
        """Loader end hook.

        Args:
            state (State): current state
        """
        self.features = np.stack(self.features)
        self.targets = np.stack(self.targets)

        if len(np.unique(self.targets)) > self.num_classes:
            raise Warning("Targets has more classes than num_classes")

        s = {
            "values": self.features,
            "labels": self.targets,
        }

        self.sets[state.loader_name] = s

        y_true, y_pred = self._knn(s)

        loader_values = state.loader_metrics
        if self.num_classes == 2:
            loader_values[self.prefix] = self.metric_fn(
                y_true, y_pred, average="binary"
            )
        else:
            values = self.metric_fn(y_true, y_pred, average=None)

            loader_values[f"{self.prefix}"] = np.mean(values)
            for i, value in enumerate(values):
                loader_values[f"{self.prefix}/{self.class_names[i]}"] = value

        self._reset_cache()

    def on_epoch_end(self, state: State) -> None:
        """Epoch end hook.

        Args:
            state (State): current state
        """
        if self.cv_loader_names is not None:
            for k, vs in self.cv_loader_names.items():

                # checking for presence of subset
                if k not in self.sets:
                    print(
                        f"Set `{k}` not found in the sets. "
                        f"Please change `cv_loader_names` parameter."
                    )
                    continue

                for v in vs:

                    # checking for presence of subset
                    if v not in self.sets:
                        print(
                            f"Set `{v}` not found in the sets. "
                            f"Please change `cv_loader_names` parameter."
                        )
                        continue

                    y_true, y_pred = self._knn(self.sets[k], self.sets[v])

                    loader_values = state.epoch_metrics[f"{k}_{v}_cv"]

                    if self.num_classes == 2:
                        loader_values[f"{self.prefix}"] = self.metric_fn(
                            y_true, y_pred, average="binary"
                        )
                    else:
                        values = self.metric_fn(y_true, y_pred, average=None)

                        loader_values[f"{self.prefix}"] = np.mean(values)
                        for i, value in enumerate(values):
                            prefix = f"{self.prefix}/{self.class_names[i]}"
                            loader_values[prefix] = value

        self._reset_cache()
        self._reset_sets()


__all__ = ["KNNMetricCallback"]
