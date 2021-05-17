from typing import List, Union

import torch

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._accuracy import AccuracyMetric, MultilabelAccuracyMetric


class AccuracyCallback(BatchMetricCallback):
    """Accuracy metric callback.
    Computes multiclass accuracy@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        topk_args: specifies which accuracy@K to log:
            [1] - accuracy
            [1, 3] - accuracy at 1 and 3
            [1, 3, 5] - accuracy at 1, 3 and 5
        num_classes: number of classes to calculate ``topk_args`` if ``accuracy_args`` is None
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
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
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        num_classes: int = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=AccuracyMetric(
                topk_args=topk_args, num_classes=num_classes, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MultilabelAccuracyCallback(BatchMetricCallback):
    """Multilabel accuracy metric callback.
    Computes multilabel accuracy@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        threshold: thresholds for model scores
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

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
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AUCCallback(input_key="logits", target_key="targets"),
                dl.MultilabelAccuracyCallback(
                    input_key="logits", target_key="targets", threshold=0.5
                )
            ]

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        threshold: Union[float, torch.Tensor] = 0.5,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MultilabelAccuracyMetric(threshold=threshold, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["AccuracyCallback", "MultilabelAccuracyCallback"]
