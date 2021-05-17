from typing import Any, Callable, Dict, List, TYPE_CHECKING, Union

import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


def _sum_aggregation(x):
    return torch.sum(torch.stack(x))


def _mean_aggregation(x):
    return torch.mean(torch.stack(x))


class MetricAggregationCallback(Callback):
    """A callback to aggregate several metrics in one value.

    Args:
        metric_key: new key for aggregated metric.
        metrics (Union[str, List[str], Dict[str, float]]): If not None,
            it aggregates only the values from the metric by these keys.
            for ``weighted_sum`` aggregation it must be a Dict[str, float].
        mode: function for aggregation.
            Must be either ``sum``, ``mean`` or ``weighted_sum`` or user's
            function to aggregate metrics. This function must get dict of
            metrics and runner and return aggregated metric. It can be
            useful for complicated fine tuning with different losses that
            depends on epochs and loader or something also
        scope: type of metric. Must be either ``batch`` or ``loader``
        multiplier: scale factor for the aggregated metric.

    Python example - loss is a weighted sum of cross entropy loss and binary cross entropy loss:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from catalyst import dl

        # data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = {"ce": torch.nn.CrossEntropyLoss(), "bce": torch.nn.BCEWithLogitsLoss()}
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # runner
        class CustomRunner(dl.Runner):
            def handle_batch(self, batch):
                x, y = batch
                logits = self.model(x)
                num_classes = logits.shape[-1]
                targets_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes)
                self.batch = {
                    "features": x,
                    "logits": logits,
                    "targets": y,
                    "targets_onehot": targets_onehot.float(),
                }


        # training
        runner = CustomRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            num_epochs=3,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.CriterionCallback(
                    input_key="logits",
                    target_key="targets",
                    metric_key="loss_ce",
                    criterion_key="ce",
                ),
                dl.CriterionCallback(
                    input_key="logits",
                    target_key="targets_onehot",
                    metric_key="loss_bce",
                    criterion_key="bce",
                ),
                # loss aggregation
                dl.MetricAggregationCallback(
                    metric_key="loss",
                    metrics={"loss_ce": 0.6, "loss_bce": 0.4},
                    mode="weighted_sum",
                ),
                dl.OptimizerCallback(metric_key="loss"),
            ],
        )

    """

    def __init__(
        self,
        metric_key: str,
        metrics: Union[str, List[str], Dict[str, float]] = None,
        mode: Union[str, Callable] = "mean",
        scope: str = "batch",
        multiplier: float = 1.0,
    ) -> None:
        """Init."""
        super().__init__(order=CallbackOrder.metric_aggregation, node=CallbackNode.all)

        if metric_key is None or not isinstance(metric_key, str):
            raise ValueError("prefix must be str")

        if mode in ("sum", "mean"):
            if metrics is not None and not isinstance(metrics, list):
                raise ValueError(
                    "For `sum` or `mean` mode the metrics must be "
                    "None or list or str (not dict)"
                )
        elif mode in ("weighted_sum", "weighted_mean"):
            if metrics is None or not isinstance(metrics, dict):
                raise ValueError(
                    "For `weighted_sum` or `weighted_mean` mode "
                    "the metrics must be specified "
                    "and must be a dict"
                )
        elif not callable(mode):
            raise NotImplementedError(
                "mode must be `sum`, `mean` " "or `weighted_sum` or `weighted_mean` or be Callable"
            )

        assert scope in ("batch", "loader")

        if isinstance(metrics, str):
            metrics = [metrics]

        self.metric_key = metric_key
        self.metrics = metrics
        self.mode = mode
        self.scope = scope
        self.multiplier = multiplier

        if mode in ("sum", "weighted_sum", "weighted_mean"):
            self.aggregation_fn = _sum_aggregation
            if mode == "weighted_mean":
                weights_sum = sum(metrics.items())
                self.metrics = {key: weight / weights_sum for key, weight in metrics.items()}
        elif mode == "mean":
            self.aggregation_fn = _mean_aggregation
        elif callable(mode):
            self.aggregation_fn = mode

    def _preprocess(self, metrics: Any) -> List[float]:
        if self.metrics is not None:
            try:
                if self.mode == "weighted_sum":
                    result = [metrics[key] * value for key, value in self.metrics.items()]
                else:
                    result = [metrics[key] for key in self.metrics]
            except KeyError:
                raise KeyError(f"Could not found required key out of {metrics.keys()}")
        else:
            result = list(metrics.values())
        result = [metric.float() for metric in result]
        return result

    def _process_metrics(self, metrics: Dict, runner: "IRunner") -> None:
        if callable(self.mode):
            metric_aggregated = self.aggregation_fn(metrics, runner) * self.multiplier
        else:
            metrics_processed = self._preprocess(metrics)
            metric_aggregated = self.aggregation_fn(metrics_processed) * self.multiplier
        metrics[self.metric_key] = metric_aggregated

    def on_batch_end(self, runner: "IRunner") -> None:
        """Computes the metric and add it to the batch metrics.

        Args:
            runner: current runner
        """
        if self.scope == "batch":
            self._process_metrics(runner.batch_metrics, runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """Computes the metric and add it to the loader metrics.

        Args:
            runner: current runner
        """
        self._process_metrics(runner.loader_metrics, runner)


__all__ = [
    "MetricAggregationCallback",
]
