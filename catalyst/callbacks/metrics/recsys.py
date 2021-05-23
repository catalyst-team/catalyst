from typing import List

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._hitrate import HitrateMetric
from catalyst.metrics._map import MAPMetric
from catalyst.metrics._mrr import MRRMetric
from catalyst.metrics._ndcg import NDCGMetric


class HitrateCallback(BatchMetricCallback):
    """Hitrate metric callback.
    Computes  HR@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        topk_args: specifies which HR@K to log:
            [1] - HR
            [1, 3] - HR at 1 and 3
            [1, 3, 5] - HR at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
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
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.CriterionCallback(
                    input_key="logits", target_key="targets", metric_key="loss"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(
                    input_key="scores", target_key="targets", topk_args=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
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
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=HitrateMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MAPCallback(BatchMetricCallback):
    """MAP metric callback.
    Computes  MAP@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which MAP@K to log:
            [1] - MAP
            [1, 3] - MAP at 1 and 3
            [1, 3, 5] - MAP at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
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
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.CriterionCallback(
                    input_key="logits", target_key="targets", metric_key="loss"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(
                    input_key="scores", target_key="targets", topk_args=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
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
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MAPMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MRRCallback(BatchMetricCallback):
    """MRR metric callback.
    Computes  MRR@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which MRR@K to log:
            [1] - MRR
            [1, 3] - MRR at 1 and 3
            [1, 3, 5] - MRR at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
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
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.CriterionCallback(
                    input_key="logits", target_key="targets", metric_key="loss"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(
                    input_key="scores", target_key="targets", topk_args=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
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
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MRRMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class NDCGCallback(BatchMetricCallback):
    """NDCG metric callback.
    Computes  NDCG@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which NDCG@K to log:
            [1] - NDCG
            [1, 3] - NDCG at 1 and 3
            [1, 3, 5] - NDCG at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
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
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.CriterionCallback(
                    input_key="logits", target_key="targets", metric_key="loss"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(
                    input_key="scores", target_key="targets", topk_args=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
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
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=NDCGMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["HitrateCallback", "MAPCallback", "MRRCallback", "NDCGCallback"]
