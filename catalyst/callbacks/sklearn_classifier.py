from typing import Callable

from catalyst.core import CallbackOrder, IRunner
from catalyst.core.callback import Callback
from catalyst.metrics._metric import AccumulationMetric


class SklearnClassifierCallback(Callback):
    """Callback to train a classifier on the train loader and
    give a metric score on the valid loader.

    Args:
        feature_key: keys of tensors that should be used as features in the classifier calculations
        target_key: keys of tensors that should be used as targets in the classifier calculations
        train_loader: train loader name
        valid_loader: valid loader name
        sklearn_classifier: fabric to produce objects with .fit and .predict_proba methods
        sklearn_metric_fn: metric to calculate in callback at the valid loader end

    Examples:
        code-block:: python
            from functools import partial
            import os

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import top_k_accuracy_score
            from torch.optim import Adam
            from torch.utils.data import DataLoader

            from catalyst import data, dl
            from catalyst.contrib import datasets, models, nn
            from catalyst.data.transforms import Compose, Normalize, ToTensor

            # 1. train and valid loaders
            transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MnistMLDataset(
                root=os.getcwd(),
                download=True,
                transform=transforms
            )
            sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
            train_loader = DataLoader(
                dataset=train_dataset,
                sampler=sampler,
                batch_size=sampler.batch_size)

            valid_dataset = datasets.MNIST(root=os.getcwd(), transform=transforms, train=False)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

            # 2. model and optimizer
            model = models.MnistSimpleNet(out_features=16)
            optimizer = Adam(model.parameters(), lr=0.001)

            # 3. criterion with triplets sampling
            sampler_inbatch = data.HardTripletsSampler(norm_required=False)
            criterion = nn.TripletMarginLossWithSampler(
                margin=0.5,
                sampler_inbatch=sampler_inbatch
            )

            # 4. training with catalyst Runner
            class CustomRunner(dl.SupervisedRunner):
                def handle_batch(self, batch) -> None:
                    images, targets = batch["features"].float(), batch["targets"].long()
                    features = self.model(images)
                    self.batch = {
                        "embeddings": features,
                        "targets": targets,
                    }

            callbacks = [
                dl.ControlFlowCallback(
                    dl.CriterionCallback(
                        input_key="embeddings",
                        target_key="targets",
                        metric_key="loss"),
                    loaders="train",
                ),
                dl.SklearnClassifierCallback(
                    feature_key="embeddings",
                    target_key="targets",
                    train_loader="train",
                    valid_loader="valid",
                    sklearn_classifier_fn=LogisticRegression,
                    sklearn_metric_fn=partial(top_k_accuracy_score, **{"k": 1}),
                ),
            ]

            runner = CustomRunner(input_key="features", output_key="embeddings")
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                callbacks=callbacks,
                loaders={"train": train_loader, "valid": valid_loader},
                verbose=False,
                logdir="./logs",
                valid_loader="valid",
                valid_metric="sklearn_classifier_metric",
                minimize_valid_metric=False,
                num_epochs=100,
            )
    """

    def __init__(
        self,
        feature_key: str,
        target_key: str,
        train_loader: str,
        valid_loader: str,
        sklearn_classifier_fn: Callable,
        sklearn_metric_fn: Callable,
    ) -> None:
        super().__init__(order=CallbackOrder.Metric)
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self.classifier_fabric = sklearn_classifier_fn
        self.metric_fn = sklearn_metric_fn
        self.feature_key = feature_key
        self.target_key = target_key
        self.storage = {
            self._train_loader: AccumulationMetric(accumulative_fields=[feature_key, target_key]),
            self._valid_loader: AccumulationMetric(accumulative_fields=[feature_key, target_key]),
        }
        self.classifier = None

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Loader start hook: initiliaze storages for the loaders.

        Args:
            runner: current runner
        """
        super().on_loader_start(runner)
        if runner.loader_key in [self._train_loader, self._valid_loader]:
            self.storage[runner.loader_key].reset(
                num_samples=runner.loader_batch_size * runner.loader_batch_len,
                num_batches=runner.loader_batch_len,
            )
        if runner.loader_key == self._valid_loader:
            assert self.classifier is not None, "The train loader has to be processed first!"

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action: get data from runner's batch and update a loader storage with it

        Args:
            runner: runner for the experiment.
        """
        if runner.loader_key in self.storage:
            loader_storage = self.storage[runner.loader_key]
            loader_storage.update(**runner.batch)

    def on_loader_end(self, runner: "IRunner") -> None:
        """
        Loader end hook: for the train loader train classifier/for the test check the quality

        Args:
            runner: current runner
        """
        if runner.loader_key == self._train_loader:
            data = self.storage[self._train_loader].compute_key_value()
            # classifier fit
            features, targets = data[self.feature_key].numpy(), data[self.target_key].numpy()
            self.classifier = self.classifier_fabric()
            self.classifier.fit(features, targets)
        if runner.loader_key == self._valid_loader:
            data = self.storage[self._train_loader].compute_key_value()
            features, y_true = data[self.feature_key], data[self.target_key]
            # classifier predict
            y_pred = self.classifier.predict_proba(features)
            metric_val = self.metric_fn(y_true, y_pred)
            runner.loader_metrics.update({"sklearn_classifier_metric": metric_val})
            # delete classifier
            self.classifier = None


__all__ = ["SklearnClassifierCallback"]
