from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics._r2_squared import R2Squared


class R2SquaredCallback(LoaderMetricCallback):
    """R2 Squared  metric callback.

    Args:
        input_key: input key to use for r2squared calculation, specifies our ``y_true``.
        target_key: output key to use for r2squared calculation, specifies our ``y_pred``.
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # data
        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        # model training
        runner = dl.SupervisedRunner()
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
            num_epochs=8,
            verbose=True,
            callbacks=[
                      dl.R2SquaredCallback(input_key="logits", target_key="targets")
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=R2Squared(prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
        )


__all__ = ["R2SquaredCallback"]
