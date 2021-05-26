from typing import Any, Callable, Dict, Mapping, Union
from functools import partial

import torch

from catalyst.callbacks.metric import FunctionalBatchMetricCallback, LoaderMetricCallback
from catalyst.core.runner import IRunner
from catalyst.metrics import FunctionalBatchMetric, FunctionalLoaderMetric
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    import sklearn


class SklearnBatchCallback(FunctionalBatchMetricCallback):
    """SklearnBatchCallback implements an integration of **batch-based** Sklearn metrics

    Args:
        keys: a dictionary containing:
            a mapping between ``metric_fn`` arguments and keys in ``runner.batch``
            other arguments needed for ``metric_fn``
        metric_fn: metric function that gets outputs, targets, and other arguments given
            in ``keys`` and returns score
        metric_key: key to store computed metric in ``runner.batch_metrics`` dictionary
        log_on_batch: boolean flag to log computed metrics every batch
        metric_kwargs: additional parameters for ``metric_fn``

    .. note::
        catalyst[ml] required for this callback

    Examples:

    .. code-block:: python

        import sklearn
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl
        from functools import partial

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=3,
            verbose=True,
            callbacks=[
                dl.BatchTransformCallback(
                    input_key="targets",
                    output_key="labels",
                    transform=partial(torch.argmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.BatchTransformCallback(
                    input_key="logits",
                    output_key="scores",
                    transform=partial(torch.softmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.BatchTransformCallback(
                    input_key="scores",
                    output_key="preds",
                    transform=partial(torch.argmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.MultilabelAccuracyCallback(
                    input_key="logits", target_key="targets", threshold=0.5
                ),
                dl.SklearnBatchCallback(
                    keys={"y_pred": "preds", "y_true": "labels"},
                    metric_fn="f1_score",
                    metric_key="sk_f1",
                    average="macro",
                    zero_division=1,
                )
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        keys: Mapping[str, Any],
        metric_fn: Union[Callable, str],
        metric_key: str,
        log_on_batch: bool = True,
        **metric_kwargs
    ):
        """Init."""
        if isinstance(metric_fn, str):
            metric_fn = sklearn.metrics.__dict__[metric_fn]
        metric_fn = partial(metric_fn, **metric_kwargs)

        super().__init__(
            metric=FunctionalBatchMetric(metric_fn=metric_fn, metric_key=metric_key),
            input_key=keys,
            target_key=keys,
            log_on_batch=log_on_batch,
        )

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """Get data from batch in a format suitable for the Sklearn metrics calculation

        Args:
            runner: current runner

        Returns:
            dict of inputs and targets tensors
        """
        kv_inputs = {}
        for key, value in self._keys.items():
            kv_inputs[key] = runner.batch[value].cpu().detach()
        kv_inputs["batch_size"] = runner.batch_size
        return kv_inputs


class SklearnLoaderCallback(LoaderMetricCallback):
    """SklearnLoaderCallback implements an integration of **loader-based** Sklearn metrics

    Args:
        keys: a mapping between ``metric_fn`` arguments and keys in ``runner.batch``
        metric_fn: metric function that gets outputs, targets, and other arguments given
            in ``keys`` and returns score
        metric_key: key to store computed metric in ``runner.batch_metrics`` dictionary
        metric_kwargs: additional parameters for ``metric_fn``

    .. note::
        catalyst[ml] required for this callback

    Examples:

    .. code-block:: python

        import sklearn
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl
        from functools import partial

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=3,
            verbose=True,
            callbacks=[
                dl.BatchTransformCallback(
                    input_key="targets",
                    output_key="labels",
                    transform=partial(torch.argmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.BatchTransformCallback(
                    input_key="logits",
                    output_key="scores",
                    transform=partial(torch.softmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.BatchTransformCallback(
                    input_key="scores",
                    output_key="preds",
                    transform=partial(torch.argmax, dim=1),
                    scope="on_batch_end",
                ),
                dl.MultilabelAccuracyCallback(
                    input_key="logits", target_key="targets", threshold=0.5
                ),
                dl.SklearnLoaderCallback(
                    keys={"y_score": "scores", "y_true": "labels"},
                    metric_fn="roc_auc_score",
                    metric_key="roc_auc_score",
                    average="macro",
                    multi_class="ovo"
                )
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        keys: Mapping[str, Any],
        metric_fn: Union[Callable, str],
        metric_key: str,
        **metric_kwargs
    ):
        """Init."""
        if isinstance(metric_fn, str):
            metric_fn = sklearn.metrics.__dict__[metric_fn]
        metric_fn = partial(metric_fn, **metric_kwargs)

        super().__init__(
            metric=FunctionalLoaderMetric(
                metric_fn=metric_fn, metric_key=metric_key, accumulative_fields=keys
            ),
            input_key=keys,
            target_key=keys,
        )

    def _get_key_value_inputs(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """Get data from batch in a format suitable for the Sklearn metrics calculation

        Args:
            runner: current runner

        Returns:
            dict of inputs and targets tensors
        """
        kv_inputs = {}
        for key, value in self._keys.items():
            kv_inputs[key] = runner.batch[value].cpu().detach()
        return kv_inputs


__all__ = ["SklearnBatchCallback", "SklearnLoaderCallback"]
