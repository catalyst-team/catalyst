from catalyst.callbacks.metrics.functional_metric import FunctionalMetricCallback
from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner
from catalyst.utils.misc import get_attr


class ICriterionCallback(Callback):
    """Criterion callback interface, abstraction over criterion step."""

    pass


class CriterionCallback(FunctionalMetricCallback, ICriterionCallback):
    """Criterion callback, abstraction over criterion step.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        metric_key: key to store computed metric in ``runner.batch_metrics`` dictionary
        criterion_key: A key to take a criterion in case
            there are several of them, and they are in a dictionary format.

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
        metric_key: str,
        criterion_key: str = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            input_key=input_key,
            target_key=target_key,
            metric_fn=self._metric_fn,
            metric_key=metric_key,
            compute_on_call=True,
            log_on_batch=True,
            prefix=prefix,
            suffix=suffix,
        )
        self.criterion_key = criterion_key
        self.criterion = None

    def _metric_fn(self, *args, **kwargs):
        return self.criterion(*args, **kwargs)

    def on_stage_start(self, runner: "IRunner"):
        """Checks that the current stage has correct criterion.

        Args:
            runner: current runner
        """
        self.criterion = get_attr(runner, key="criterion", inner_key=self.criterion_key)
        assert self.criterion is not None


__all__ = ["ICriterionCallback", "CriterionCallback"]
