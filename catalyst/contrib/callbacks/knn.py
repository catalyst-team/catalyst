from catalyst.dl.core import Callback, RunnerState, CallbackOrder

import torch
import numpy as np

from math import ceil
from scipy import stats

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score, \
    precision_score, recall_score


class KNNMetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """

    def __init__(
            self,
            input_key: str = "logits",
            output_key: str = "targets",
            prefix: str = "knn",
            num_classes: int = 2,
            class_names: dict = None,
            metric_fn: str = "f1-score",
            knn_metric: str = "euclidean",
            n_neighbors: int = 5
    ):
        """Returns accuracy calculated using kNN algorithm.
        Args:
            input_key (str): input key to get features
            output_key (str): output key to get targets
            prefix (str): key to store in logs
            num_classes (int): Number of classes; must be > 1
            class_names (dict): of indexes and class names
            metric_fn (str): one of `accuracy`, `precision`, `recall`, 
                            `f1-score`, default is `f1-score`
            knn_metric (str): look sklearn.neighbors.NearestNeighbors parameter
            n_neighbors (int): number of neighbors
        """

        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.features_key = input_key
        self.targets_key = output_key

        self.num_classes = num_classes
        self.class_names = class_names

        if metric_fn == 'accuracy':
            self.metric_fn = accuracy_score
        elif metric_fn == 'recall':
            self.metric_fn = recall_score
        elif metric_fn == 'precision':
            self.metric_fn = precision_score
        elif metric_fn == 'f1-score':
            self.metric_fn = f1_score
        else:
            raise ValueError(f"Metric function with value '{metric_fn}'"
                             f" not implemented")

        self.knn_metric = knn_metric
        self.n_neighbors = n_neighbors

        self.n_folds = 1

        self.reset_cache()
        self.reset_sets()

    def on_batch_end(self, state: RunnerState):
        features: torch.Tensor = \
            state.output[self.features_key].cpu().detach().numpy()
        targets: torch.Tensor = \
            state.input[self.targets_key].cpu().detach().numpy()

        self.features.extend(features)
        self.targets.extend(targets)

    def on_loader_end(self, state: RunnerState):

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

        loader_values = state.metrics.epoch_values[state.loader_name]
        if self.num_classes == 2:
            loader_values[self.prefix] = \
                self.metric_fn(y_true, y_pred, average="binary")
        else:
            values = self.metric_fn(y_true, y_pred, average=None)

            loader_values[f"{self.prefix}"] = np.mean(values)
            for i, value in enumerate(values):
                postfix = self.class_names[i] \
                    if self.class_names is not None else str(i)
                loader_values[f"{self.prefix}/{postfix}"] = value

        self.reset_cache()

    def on_epoch_end(self, state: RunnerState):

        for s in ["valid", "test"]:
            if s in self.sets:
                y_true, y_pred = self._knn(self.sets["train"], self.sets[s])

                loader_values = \
                    state.metrics.epoch_values[f"train_{s}_cv"]
                if self.num_classes == 2:
                    loader_values[f"{self.prefix}"] = \
                        self.metric_fn(y_true, y_pred, average="binary")
                else:
                    values = self.metric_fn(y_true, y_pred, average=None)

                    loader_values[f"{self.prefix}"] = np.mean(values)
                    for i, value in enumerate(values):
                        postfix = self.class_names[i] \
                            if self.class_names is not None else str(i)
                        loader_values[f"{self.prefix}/{postfix}"] = \
                            value

        self.reset_cache()
        self.reset_sets()

    def reset_cache(self):
        self.features = []
        self.targets = []

    def reset_sets(self):
        self.sets = {}

    def _knn(self, train_set, test_set=None):
        """Returns accuracy calculated using kNN algorithm.
        Args:
            train_set: dict of feature "values" and "labels" for training set.
            test_set: dict of feature "values" and "labels" for test set.
        Returns:
            cm: np.ndarray of confusion matrix
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
                    n_neighbors=self.n_neighbors + int(leave_one_out),
                    metric=self.knn_metric,
                    algorithm="brute")
                classifier.fit(x_train, y_train)

                # data could be evaluated in n_folds in order to avoid OOM
                end_idx, batch_size = 0, ceil(size / self.n_folds)
                for s, start_idx in enumerate(range(0, size, batch_size)):

                    end_idx = min(start_idx + batch_size, size)

                    x = x_test[start_idx: end_idx]

                    knn_ids = classifier.kneighbors(x, return_distance=False)

                    # if we predict train set on itself we have to delete 0th
                    # neighbor for all of the distances
                    if leave_one_out:
                        knn_ids = knn_ids[:, 1:]

                    knn_classes = y_train[knn_ids]
                    knn_classes, _ = stats.mode(knn_classes, axis=1)

                    y_pred.extend(knn_classes[:, 0].tolist())

                y_pred = np.asarray(y_pred)

                result = (y_test, y_pred)

            # this try catch block made because sometimes sets are quite big
            # and it is not possible to put everything in memory, so we split
            except MemoryError:
                print(f"Memory error with {self.n_folds} fold, trying more.")
                self.n_folds *= 2
                result = None

        return result


__all__ = ["KNNMetricCallback"]
