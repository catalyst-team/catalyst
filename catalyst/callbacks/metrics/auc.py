from typing import TYPE_CHECKING

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics._auc import AUCMetric
from catalyst.settings import SETTINGS

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class AUCCallback(LoaderMetricCallback):
    """ROC-AUC  metric callback.

    Args:
        input_key: input key to use for auc calculation, specifies our ``y_true``.
        target_key: output key to use for auc calculation, specifies our ``y_pred``.
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
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=AUCMetric(
                compute_per_class_metrics=compute_per_class_metrics,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
        )

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        assert (
            not runner.engine.use_fp16
        ), "AUCCallback could not work within amp training"
        return super().on_experiment_start(runner)


__all__ = ["AUCCallback"]
