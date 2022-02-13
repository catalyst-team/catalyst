from typing import Optional

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._classification import (
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)
from catalyst.settings import SETTINGS


class PrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multiclass PrecisionRecallF1Support metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        num_classes: number of classes
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        log_on_batch: boolean flag to log computed metrics every batch
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
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
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
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

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: Optional[int] = None,
        zero_division: int = 0,
        log_on_batch: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MulticlassPrecisionRecallF1SupportMetric(
                zero_division=zero_division,
                prefix=prefix,
                suffix=suffix,
                compute_per_class_metrics=compute_per_class_metrics,
                num_classes=num_classes,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MultilabelPrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multilabel PrecisionRecallF1Support metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        num_classes: number of classes
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        log_on_batch: boolean flag to log computed metrics every batch
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
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
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
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
                dl.BatchTransformCallback(
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.MultilabelAccuracyCallback(
                    input_key="scores", target_key="targets", threshold=0.5
                ),
                dl.MultilabelPrecisionRecallF1SupportCallback(
                    input_key="scores", target_key="targets", num_classes=num_classes
                ),
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: Optional[int] = None,
        zero_division: int = 0,
        log_on_batch: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MultilabelPrecisionRecallF1SupportMetric(
                zero_division=zero_division,
                prefix=prefix,
                suffix=suffix,
                compute_per_class_metrics=compute_per_class_metrics,
                num_classes=num_classes,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = [
    "PrecisionRecallF1SupportCallback",
    "MultilabelPrecisionRecallF1SupportCallback",
]
