from typing import Iterable

from catalyst.metrics._topk_metric import TopKMetric
from catalyst.metrics.functional._hitrate import hitrate


class HitrateMetric(TopKMetric):
    """Calculates the hitrate.

    Args:
        topk: list of `topk` for hitrate@topk computing
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

    Compute mean value of hitrate and it's approximate std value.

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        outputs = torch.Tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
        targets = torch.Tensor([[0, 0, 1.0, 1.0], [0, 0, 0.0, 0.0]])
        metric = metrics.HitrateMetric(topk=[1, 2, 3, 4])
        metric.reset()

        metric.update(outputs, targets)
        metric.compute()
        # (
        #     (0.0, 0.25, 0.25, 0.5),  # mean for @01, @02, @03, @04
        #     (0.0, 0.0, 0.0, 0.0)     # std for @01, @02, @03, @04
        # )

        metric.compute_key_value()
        # {
        #     'hitrate01': 0.0,
        #     'hitrate01/std': 0.0,
        #     'hitrate02': 0.25,
        #     'hitrate02/std': 0.0,
        #     'hitrate03': 0.25,
        #     'hitrate03/std': 0.0,
        #     'hitrate04': 0.5,
        #     'hitrate04/std': 0.0
        # }

        metric.reset()
        metric(outputs, targets)
        # (
        #     (0.0, 0.25, 0.25, 0.5),  # mean for @01, @02, @03, @04
        #     (0.0, 0.0, 0.0, 0.0)     # std for @01, @02, @03, @04
        # )

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
                    input_key="scores", target_key="targets", topk=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk=(1, 3)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
            ]
        )

    .. note::
        Metric names depending on input parameters:

        - ``topk = (1,) or None`` ---> ``"hitrate01"``
        - ``topk = (1, 3)`` ---> ``"hitrate01"``, ``"hitrate03"``
        - ``topk = (1, 3, 5)`` ---> ``"hitrate01"``, ``"hitrate03"``, ``"hitrate05"``

        You can find them in ``runner.batch_metrics``, ``runner.loader_metrics`` or
        ``runner.epoch_metrics``.

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        topk: Iterable[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init HitrateMetric"""
        super().__init__(
            metric_name="hitrate",
            metric_function=hitrate,
            topk=topk,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
        )


__all__ = ["HitrateMetric"]
